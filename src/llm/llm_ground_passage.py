"""
特許審査の段階的進歩性判断システム (統合版)
幹（Claim 1）と枝葉（Claim 2以降）を段階的に検証

【統合された特徴】
- データクラスによる型安全性 (llm_pipeline.py)
- チャットセッション方式による文脈保持 (llm_pipline_gemini.py)
- 堅牢なJSONパース処理 (llm_pipeline_chatgpt.py)
- プロンプトテンプレートの外部化 (llm_pipline_gemini.py)
- 詳細な進捗表示と結果保存 (llm_pipeline.py)
"""

import google.generativeai as genai
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from dotenv import load_dotenv
import time
from google.api_core import exceptions as google_exceptions
import re

import json
import re

# ==========================================
# 1. Evidence Extraction System Class
# ==========================================

@dataclass
class ExtractionResult:
    """証拠抽出結果を保持するデータクラス"""
    target_concept: Optional[str] = None
    search_keywords: Optional[List[str]] = None
    candidates: Optional[List[Dict]] = None
    verified_evidence: Optional[List[Dict]] = None
    error: Optional[str] = None


class EvidenceExtractionSystem:
    """証拠抽出システム（Google Gemini API使用）"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Args:
            api_key: Google AI Studio APIキー
            model_name: 使用するGeminiモデル
        """
        if not api_key:
            raise ValueError("APIキーが設定されていません。.envファイルを確認してください。")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # JSON出力用のモデル
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        JSONレスポンスを堅牢にパース

        Args:
            response_text: レスポンステキスト

        Returns:
            パースされたJSON辞書
        """
        try:
            result = json.loads(response_text)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            return result
        except json.JSONDecodeError:
            # マークダウンのコードブロックを除去して再試行
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                return json.loads(response_text.strip())

    def _generate_with_retry(self, use_json_model: bool, prompt: str,
                            max_retries: int = 5, initial_wait: int = 2) -> str:
        """
        リトライロジック付きでコンテンツを生成

        Args:
            use_json_model: JSON出力モデルを使用するか
            prompt: プロンプト
            max_retries: 最大リトライ回数
            initial_wait: 初期待機時間（秒）

        Returns:
            レスポンステキスト
        """
        model = self.json_model if use_json_model else self.model

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except google_exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = initial_wait * (4 ** attempt)
                    print(f"\n⏳ レート制限エラー。{wait_time}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ 最大リトライ回数に達しました。エラー: {e}")
                    raise
            except Exception as e:
                print(f"\n❌ 予期しないエラー: {e}")
                raise

    def run_extraction_workflow(self, review_json: List[Dict], patent_json: Dict) -> Dict:
        """
        3段階のプロセスを実行するメイン関数

        Args:
            review_json: 拒絶理由のリスト
            patent_json: 先行技術文献のJSON

        Returns:
            証拠抽出結果の辞書
        """
        # 処理履歴を格納するリスト
        extraction_history = []

        try:
            print("=" * 80)
            print("📋 証拠抽出ワークフロー開始")
            print("=" * 80)

            # -------------------------------------------------
            # 準備: データのロードと整形
            # -------------------------------------------------

            # 拒絶理由の取得
            target_review = review_json[0]
            rejection_arg = target_review["examiner_review"]
            claim_element = str(target_review["application_structure"]["claim1_requirements"])

            # 文献番号の取得
            doc_number = patent_json.get("doc_number", "Unknown")

            # 先行技術文献のテキスト整形
            flattened_paragraphs = flatten_patent_description(patent_json["description"])
            full_text_str = json.dumps(flattened_paragraphs[:50], ensure_ascii=False)

            # -------------------------------------------------
            # Step 1: 論点抽出
            # -------------------------------------------------
            print("\n" + "=" * 80)
            print("🔍 Step 1: 論点抽出")
            print("=" * 80)

            prompt_1 = PROMPT_STEP_1.format(
                rejection_argument=rejection_arg,
                claim_element=claim_element
            )

            response_1 = self._generate_with_retry(use_json_model=True, prompt=prompt_1)
            result_1 = self._parse_json_response(response_1)

            if not result_1:
                print("❌ Step 1 failed.")
                return {
                    "error": "Step 1 failed",
                    "doc_number": doc_number,
                    "extraction_history": extraction_history,
                    "partial_results": "Step 1で処理が中断されました"
                }

            print("\n✅ Step 1 完了:")
            print(f"  概念: {result_1.get('target_concept')}")
            print(f"  キーワード: {result_1.get('search_keywords')}")

            extraction_history.append({
                "step": "1",
                "role": "論点抽出",
                "content": result_1
            })

            # -------------------------------------------------
            # Step 2: 候補特定
            # -------------------------------------------------
            print("\n" + "=" * 80)
            print("🔍 Step 2: 候補特定")
            print("=" * 80)

            prompt_2 = PROMPT_STEP_2.format(
                target_concept=result_1.get("target_concept"),
                search_keywords=result_1.get("search_keywords"),
                full_text_paragraphs=full_text_str
            )

            response_2 = self._generate_with_retry(use_json_model=True, prompt=prompt_2)
            result_2 = self._parse_json_response(response_2)

            if not result_2:
                print("❌ Step 2 failed.")
                return {
                    "error": "Step 2 failed",
                    "doc_number": doc_number,
                    "extraction_history": extraction_history,
                    "partial_results": "Step 2で処理が中断されました",
                    "step1_result": result_1
                }

            candidates = result_2.get("candidates", [])
            print(f"\n✅ Step 2 完了: {len(candidates)}個の候補を発見")

            extraction_history.append({
                "step": "2",
                "role": "候補特定",
                "content": result_2
            })

            # -------------------------------------------------
            # Step 3: エビデンス確定
            # -------------------------------------------------
            print("\n" + "=" * 80)
            print("🔍 Step 3: エビデンス確定")
            print("=" * 80)

            prompt_3 = PROMPT_STEP_3.format(
                rejection_argument=rejection_arg,
                candidate_paragraphs=json.dumps(candidates, ensure_ascii=False)
            )

            response_3 = self._generate_with_retry(use_json_model=True, prompt=prompt_3)
            result_3 = self._parse_json_response(response_3)

            print("\n✅ Step 3 完了: 最終エビデンス")
            print(json.dumps(result_3, indent=2, ensure_ascii=False))

            extraction_history.append({
                "step": "3",
                "role": "エビデンス確定",
                "content": result_3
            })

            print("\n" + "=" * 80)
            print("✅ 証拠抽出ワークフロー完了")
            print("=" * 80)

            return {
                "doc_number": doc_number,
                "step1_result": result_1,
                "step2_result": result_2,
                "step3_result": result_3,
                "verified_evidence": result_3.get("verified_evidence", []),
                "extraction_history": extraction_history,
                "rejection_argument": rejection_arg,
                "claim_element": claim_element
            }

        except Exception as e:
            print(f"\n❌ ワークフロー実行中にエラーが発生しました: {e}")
            return {
                "error": str(e),
                "doc_number": patent_json.get("doc_number", "Unknown"),
                "extraction_history": extraction_history,
                "partial_results": "処理が途中で中断されました"
            }


# ==========================================
# 2. Data Preprocessing Helpers
# ==========================================

def flatten_patent_description(description_dict: dict) -> list:
    """
    特許JSONのdescription（辞書形式）を、
    検索可能な段落リスト [{'id': '...', 'text': '...'}] に変換する。
    """
    paragraphs = []
    
    for section, content in description_dict.items():
        if isinstance(content, list):
            for i, text in enumerate(content):
                paragraphs.append({
                    "id": f"{section}_{i}",
                    "text": text
                })
        elif isinstance(content, str):
             paragraphs.append({
                "id": section,
                "text": content
            })
        elif isinstance(content, dict):
            # tech_problemなどさらにネストしている場合
            for sub_key, sub_content in content.items():
                if isinstance(sub_content, list):
                    for i, text in enumerate(sub_content):
                        paragraphs.append({
                            "id": f"{section}_{sub_key}_{i}",
                            "text": text
                        })
    return paragraphs

# ==========================================
# 3. Prompt Templates
# ==========================================

PROMPT_STEP_1 = """
Step 1: 論点抽出 (Logic Parser)
あなたは特許の拒絶理由を分析する専門家です。
以下の【拒絶理由】は、ある特許出願が先行技術（引用文献）に基づいて拒絶された理由を示しています。
この拒絶理由を裏付けるために、先行技術文献の中から証拠を見つける必要があります。

タスク:
1. 審査官が「先行技術に開示されている」と認定した具体的な技術要素を特定してください。
2. その技術要素が、先行技術文献の中でどのようなキーワードや表現で記載されている可能性があるか、検索用の「概念キーワード」を3つ挙げてください。

入力データ:
【拒絶理由】
{rejection_argument}

【対象となる請求項の要素】
{claim_element}

出力フォーマット（JSON）:
{{
  "target_concept": "検索すべき技術的概念（1文で）",
  "search_keywords": ["キーワード1", "キーワード2", "キーワード3"],
  "expected_context": "この概念が記載されていそうな箇所"
}}
"""

PROMPT_STEP_2 = """
Step 2: 候補特定 (Passage Retriever)
あなたは優秀な特許調査員です。
以下の【検索ターゲット】に合致する記述を、【特許文献全文】から探し出してください。

タスク:
1. 提供されたキーワードと概念に基づき、文献内で最も関連性が高い段落（Paragraph）をトップ3つ選んでください。
2. 各段落について、なぜそれが証拠になり得るかの理由を簡潔に述べてください。

入力データ:
【検索ターゲット】
概念: {target_concept}
キーワード: {search_keywords}

【特許文献全文 (抜粋/リスト)】
{full_text_paragraphs}

出力フォーマット（JSON）:
{{
  "candidates": [
    {{
      "paragraph_id": "段落ID",
      "text": "段落の全文...",
      "reason": "選定理由"
    }}
  ]
}}
"""

PROMPT_STEP_3 = """
Step 3: エビデンス確定 (Evidence Extractor)
あなたは厳格な証拠検証人です。
【拒絶理由】を裏付けるための「決定的な証拠」を、【候補段落】の中から抽出してください。

厳格なルール:
1. **原文維持**: 抽出するテキストは、元の文章と完全に一致しなければなりません。一文字も変更してはいけません。
2. **最小単位**: 段落全体ではなく、証明に必要な「文」または「フレーズ」のみを抽出してください。
3. **検証**: 抽出したテキストが、本当に拒絶理由（審査官の主張）を直接サポートしているか確認してください。

入力データ:
【拒絶理由】
{rejection_argument}

【候補段落】
{candidate_paragraphs}

出力フォーマット（JSON）:
{{
  "verified_evidence": [
    {{
      "quote": "抽出した原文のテキスト",
      "source_paragraph_id": "段落ID",
      "explanation": "この一文は、拒絶理由の『...』という指摘に対応している。"
    }}
  ]
}}
"""

# ==========================================
# 4. Entry Point Function for Evidence Extraction
# ==========================================

def evidence_extraction_entry(review_json: List[Dict], patent_json: Dict) -> Dict:
    """
    証拠抽出のエントリポイント関数

    Args:
        review_json: 拒絶理由のリスト
        patent_json: 先行技術文献のJSON

    Returns:
        証拠抽出結果の辞書
    """
    try:
        # .envファイルから環境変数を読み込む
        load_dotenv()

        # APIキーの設定（環境変数から取得）
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ .envファイルにGOOGLE_API_KEYを設定してください")
            return {"error": "API key not found"}

        # システムの初期化
        system = EvidenceExtractionSystem(api_key)

        # 証拠抽出ワークフローの実行
        results = system.run_extraction_workflow(review_json, patent_json)

        return results

    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return {"error": str(e)}


# ==========================================
# 5. Example Usage
# ==========================================

if __name__ == "__main__":
    """
    使用例: 証拠抽出システムのテスト実行
    """
    # 実際のデータを使用する場合の例
    mock_review_data = [{
        "examiner_review": "引用文献1には、酸化物半導体トランジスタを使用してデータ保持特性を向上させる技術が開示されている。",
        "application_structure": {
            "claim1_requirements": [
                "酸化物半導体トランジスタを使用すること",
                "低オフ電流特性を有すること",
                "データ保持期間を延長すること"
            ]
        }
    }]

    mock_patent_data = {
        "description": {
            "summary": ["本発明は、酸化物半導体トランジスタを用いた記憶回路に関する。"],
            "disclosure_tech_solution": ["当該OSトランジスタは、オフ状態におけるリーク電流が著しく小さいため、保持容量に保持された信号を長期間保つことができる。"],
            "best_mode": ["期間T3において、第1のトランジスタ201及び第2のトランジスタ203はオフ状態におけるリーク電流が著しく小さいため、保持容量202によって保持された入力信号INを長期間保つことができる。"]
        }
    }

    print("\n" + "🧪" * 40)
    print("証拠抽出システムのテスト実行")
    print("🧪" * 40 + "\n")

    # 証拠抽出の実行
    results = evidence_extraction_entry(mock_review_data, mock_patent_data)

    # 結果の表示
    if results and "error" not in results:
        print("\n" + "=" * 80)
        print("📊 最終結果サマリー")
        print("=" * 80)
        print(f"\n検出されたエビデンス数: {len(results.get('verified_evidence', []))}")
        for i, evidence in enumerate(results.get('verified_evidence', []), 1):
            print(f"\n[エビデンス {i}]")
            print(f"  引用: {evidence.get('quote', 'N/A')}")
            print(f"  出典: {evidence.get('source_paragraph_id', 'N/A')}")
    else:
        print(f"\n❌ エラー: {results.get('error', 'Unknown error')}")








    

# ==================== データクラス定義 ====================

@dataclass
class ClaimStructure:
    """クレーム構造を保持するデータクラス"""
    claim_number: int
    requirements: List[str]
    additional_limitations: Optional[List[str]] = None


@dataclass
class PatentDocument:
    """特許文献の構造化データ"""
    problem: str
    solution_principle: str
    claim1_requirements: List[str]
    claim2_limitations: Optional[List[str]] = None
    claim3_limitations: Optional[List[str]] = None
    abstract_hints: Optional[Dict[str, str]] = None


# ==================== プロンプトテンプレート ====================

class PromptTemplates:
    """プロンプトテンプレートを管理するクラス"""

    STEP_0_1_STRUCTURE_APPLICATION = """以下の「本願発明」のAbstractおよび全てのClaimを読み、特許判断に必要な要素を以下の形式で抽出・構造化してください。

---
Example:【構造化出力フォーマット】
以下のJSON形式で出力してください：

{{
  "problem": "課題（例：Example:ノズルプレートの機械的頑強性の向上）",
  "solution_principle": "解決原理（例：高熱安定性・特定の物性を持つ疎油性被膜の適用）",
  "claim1_requirements": [
    "要件A: （例：最高300℃で15%未満の重量損失）",
    "要件B: （例：接触角度 約50°超）",
    "要件C: （例：滑走角度 約30°未満）",
    "要件D: （例：290℃ かつ 350psiに曝露後も性能維持）"
  ],
  "claim2_limitations": [
    "（例：前記被膜がフッ素系ポリマーを含む、こと。）"
  ],
  "claim3_limitations": [
    "（例：前記被膜の膜厚が1μm～5μmである、こと。）"
  ]
}}

【本願発明】
Abstract: {abstract}

Claims: {claims_text}

JSON形式のみで回答してください。"""

    STEP_0_2_STRUCTURE_PRIOR_ART = """同様に、以下の「先行技術」のAbstractおよび全てのClaimを読み、同じ形式で構造化してください。**特にAbstractの「示唆（ヒント）」**を重要視してください。

【先行技術】
Abstract: {abstract}

{claims_text}

---
【構造化出力フォーマット】
以下のJSON形式で出力してください：

{{
  "problem": "課題（例：高温加熱による表面特性の低下防止、汚れ低減）",
  "solution_principle": "解決原理（例：熱に安定な撥油性低接着性コーティングの適用）",
  "claim1_requirements": [
    "要件X: （例：滑走角度 約30°未満）",
    "要件Y: （例：200℃に30分曝露後も性能維持）"
  ],
  "abstract_hints": {{
    "contact_angle": "（例：45°よりも大きな）",
    "temperature_range": "（例：180℃〜320℃の範囲）",
    "pressure_range": "（例：100psi〜400psiの範囲）"
  }}
}}

JSON形式のみで回答してください。"""

    STEP_1_APPLICANT_ARGUMENTS = """あなたは「本願発明」の代理人です。
先ほど構造化した2つの文献データに基づき、以下の2段階で「進歩性がある（容易に考えつけない）」という論理的な主張を構築してください。

【本願発明の構造化データ】
{app_data}

【先行技術の構造化データ】
{prior_data}

---

1. **第一の主張 (幹):**
   まず、**本願発明のClaim 1 (幹)**が、先行技術と比較して進歩性を有することを主張してください。
   *（ヒント：先行技術のClaim 1にはない要件の存在や、共通する要件の決定的な差異を強調する。）*

2. **予備的主張 (枝葉):**
   **仮に、Claim 1の進歩性が否定されたとしても**、**Claim 2の追加限定 (枝1)**や**Claim 3の追加限定 (枝2)**を先行技術に適用することは、先行技術からは動機付けがなく、容易想到ではないと主張してください。

---

以下の構造で主張を展開してください：

## 第一の主張：Claim 1の進歩性

### 1. 課題・解決原理の相違点
[本願発明と先行技術の課題・解決原理の違いを説明]

### 2. 構成要件の相違点
[Claim 1の要件と先行技術の要件の具体的な違いを列挙]

### 3. 進歩性の根拠
[なぜこの相違点が単なる最適化ではなく、進歩性を有するのかを論理的に説明]

## 予備的主張：Claim 2以降の進歩性

### Claim 2の追加限定について
[Claim 2の追加限定が先行技術から容易想到でない理由]

### Claim 3の追加限定について
[Claim 3の追加限定が先行技術から容易想到でない理由]
"""

    STEP_2_EXAMINER_REVIEW = """役割を変更します。あなたは特許庁の「審査官」です。
ステップ1の「代理人の主張」を踏まえつつ、**客観的かつ厳格に**進歩性（容易想到性）を審査してください。

【本願発明の構造化データ】
{app_data}

【先行技術の構造化データ】
{prior_data}

【代理人の主張】
{arguments}

---

### 【第1段階：Claim 1 (幹) の検証】

ステップ0の構造化データを参照し、以下の核心的な問いに**順番に**答えてください。

1.  **技術分野の関連性と動機付け（最重要）:**
    * 本願発明(A)と先行技術(B)は、**技術分野が関連**していますか？
    * 当業者が、Aの**技術課題**を解決するために、Bの技術を**適用または改良しようとする動機付け**（理由）は存在しますか？
    * **（重要：もし分野が無関係で動機付けが存在しない場合、直ちに「進歩性あり（容易想到ではない）」と結論付け、以降の比較は不要です。）**

2.  **構成要件の比較と容易性:**
    * （問1で「動機付けあり」と判断された場合）AとBの**差分（異なる点）**は何ですか？
    * その差分は、Bの技術や、その分野の**技術常識（周知技術）**に基づいて、当業者が**通常の創作能力（例：設計変更、材料の最適化、周知技術の組み合わせ）**を発揮すれば容易に到達できるものですか？

3.  **効果の予測可能性:**
    * Aの発明によって得られる効果は、Bや周知技術から**予測可能な範囲内（程度の差）**ですか？ それとも、**予測できない格別な効果**（異質な効果、顕著な効果）が生じていますか？

4.  **結論 (Claim 1):**
    * 上記1〜3を総合し、Claim 1は「容易想到である」か「容易想到ではない」か、理由と共に判断してください。

---

### 【第2段階：Claim 2以降 (枝葉) の検証】

（第1段階の回答を生成させた後、続けて指示します）

第1段階の結論に関わらず、**Claim 2以降の追加限定（枝）**についても個別に検討します。

* **Claim 2（枝1）の検討:**
    * この「追加限定」自体は、その分野の**技術常識**や**周知の選択肢**（例：よく使われる材料、一般的な工程）に過ぎませんか？
    * それをClaim 1（または先行技術）に適用することは**容易**ですか？ それにより**格別な効果**が生じますか？
    * 結論として、Claim 2は容易想到ですか？

* （同様に、Claim 3 (枝2) ... についても個別に検証してください）
"""

    STEP_3_FINAL_DECISION = """あなたは「主任審査官」です。
ステップ1の「代理人の段階的主張」とステップ2の「審査官の段階的検証（反論）」を比較検討してください。

【代理人の主張】
{arguments}

【審査官の検証・反論】
{review}

---

以下の項目について、最終的な進歩性の判断（容易想到である / 容易想到ではない）とその理由を簡潔に述べてください。

## 判断項目

### 1. Claim 1 (幹) の進歩性
**判断:** [容易想到である / 容易想到ではない]
**理由:** [簡潔に説明]

### 2. Claim 2 (枝1) の進歩性
**判断:** [容易想到である / 容易想到ではない]
**理由:** [簡潔に説明]

### 3. Claim 3 (枝2) の進歩性
**判断:** [容易想到である / 容易想到ではない]
**理由:** [簡潔に説明]

### 4. 総合結論
[例: Claim 1は先行技術の示唆に基づく単なる最適化であり進歩性なし。しかし、Claim 2の追加限定は周知技術とは言えず進歩性あり。よって、Claim 2以降のクレームは特許可能と判断する。]

---

以下のJSON形式でも出力してください：

{{
  "claim1": {{
    "inventive": true/false,
    "reason": "理由"
  }},
  "claim2": {{
    "inventive": true/false,
    "reason": "理由"
  }},
  "claim3": {{
    "inventive": true/false,
    "reason": "理由"
  }},
  "conclusion": "総合結論"
}}
"""


# ==================== メインシステムクラス ====================

class PatentExaminationSystemIntegrated:
    """統合版特許審査システム"""

    # def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
    # def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Args:
            api_key: Google AI Studio APIキー
            model_name: 使用するGeminiモデル
        """
        if not api_key:
            raise ValueError("APIキーが設定されていません。.envファイルを確認してください。")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # JSON出力用のモデル（構造化データ用）
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )

        # チャットセッション（文脈保持用）
        self.chat = None
        self.conversation_history = []

    def _parse_json_response(self, response_text: str) -> Dict:
        """
        JSONレスポンスを堅牢にパース

        Args:
            response_text: レスポンステキスト

        Returns:
            パースされたJSON辞書
        """
        try:
            result = json.loads(response_text)
            # リスト形式で返ってきた場合は最初の要素を取得
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            return result
        except json.JSONDecodeError:
            # マークダウンのコードブロックを除去して再試行
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # ```なしのコードブロックも試す
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                # 最後の手段として素のテキストをパース
                return json.loads(response_text.strip())

    def _generate_with_retry(self, use_json_model: bool, prompt: str,
                            max_retries: int = 5, initial_wait: int = 2) -> str:
        """
        リトライロジック付きでコンテンツを生成

        Args:
            use_json_model: JSON出力モデルを使用するか
            prompt: プロンプト
            max_retries: 最大リトライ回数
            initial_wait: 初期待機時間（秒）

        Returns:
            レスポンステキスト
        """
        model = self.json_model if use_json_model else self.model

        for attempt in range(max_retries):
            try:
                if self.chat and not use_json_model:
                    # チャットセッションを使用（文脈保持）
                    response = self.chat.send_message(prompt)
                else:
                    # 単発のリクエスト（JSON構造化用）
                    response = model.generate_content(prompt)
                return response.text
            except google_exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = initial_wait * (4 ** attempt)  # 指数バックオフ
                    print(f"\n⏳ レート制限エラー。{wait_time}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ 最大リトライ回数に達しました。エラー: {e}")
                    raise
            except Exception as e:
                print(f"\n❌ 予期しないエラー: {e}")
                raise

    def step0_structure_application(self, doc_dict: Dict) -> PatentDocument:
        """
        ステップ0.1: 本願発明の構造化

        Args:
            abstract: 本願発明のAbstract
            claims: 本願発明のClaimリスト

        Returns:
            構造化された本願発明データ
        """
        print("=" * 80)
        print("📋 ステップ0.1: 本願発明の構造化")
        print("=" * 80)

        abstract = doc_dict.get("abstract", "")
        claims_text = doc_dict.get("claims", "")

        prompt = PromptTemplates.STEP_0_1_STRUCTURE_APPLICATION.format(
            abstract=abstract,
            claims_text=claims_text
        )

        response_text = self._generate_with_retry(use_json_model=True, prompt=prompt)
        result = self._parse_json_response(response_text)

        print("\n✅ 構造化完了:")
        print(f"課題: {result['problem']}")
        print(f"解決原理: {result['solution_principle']}")
        print(f"Claim 1要件: {len(result['claim1_requirements'])}個")

        self.conversation_history.append({
            "step": doc_dict["step"],
            "role": "構造化",
            "content": result
        })

        return result


    def step1_applicant_arguments(self, app_data: Dict, prior_data: Dict) -> str:
        """
        ステップ1: 代理人の段階的主張

        Args:
            app_data: 本願発明の構造化データ
            prior_data: 先行技術の構造化データ

        Returns:
            代理人の主張テキスト
        """
        print("\n" + "=" * 80)
        print("⚖️ ステップ1: 代理人の段階的主張")
        print("=" * 80)

        prompt = PromptTemplates.STEP_1_APPLICANT_ARGUMENTS.format(
            app_data=json.dumps(app_data, ensure_ascii=False, indent=2),
            prior_data=json.dumps(prior_data, ensure_ascii=False, indent=2)
        )

        arguments = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 代理人の主張を生成しました")
        print("\n" + "-" * 80)
        print(arguments)
        print("-" * 80)

        self.conversation_history.append({
            "step": "1",
            "role": "代理人",
            "content": arguments
        })

        return arguments

    def step2_examiner_review(self, app_data: Dict, prior_data: Dict, arguments: str) -> str:
        """
        ステップ2: 審査官の段階的批評（7質問による検証）

        Args:
            app_data: 本願発明の構造化データ
            prior_data: 先行技術の構造化データ
            arguments: 代理人の主張

        Returns:
            審査官の検証・反論テキスト
        """
        print("\n" + "=" * 80)
        print("🔍 ステップ2: 審査官の専門的判断")
        print("=" * 80)

        prompt = PromptTemplates.STEP_2_EXAMINER_REVIEW.format(
            app_data=json.dumps(app_data, ensure_ascii=False, indent=2),
            prior_data=json.dumps(prior_data, ensure_ascii=False, indent=2),
            arguments=arguments
        )

        review = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 審査官の検証を生成しました")
        print("\n" + "-" * 80)
        print(review)
        print("-" * 80)

        self.conversation_history.append({
            "step": "2",
            "role": "審査官",
            "content": review
        })

        return review

    def step3_final_decision(self, arguments: str, review: str) -> str:
        """
        ステップ3: 主任審査官の段階的統合判断

        Args:
            arguments: 代理人の主張
            review: 審査官の検証・反論

        Returns:
            最終判断テキスト
        """
        print("\n" + "=" * 80)
        print("⚖️ ステップ3: 主任審査官の段階的統合判断")
        print("=" * 80)

        prompt = PromptTemplates.STEP_3_FINAL_DECISION.format(
            arguments=arguments,
            review=review
        )

        decision = self._generate_with_retry(use_json_model=False, prompt=prompt)

        print("\n✅ 最終判断を生成しました")
        print("\n" + "=" * 80)
        print(decision)
        print("=" * 80)

        self.conversation_history.append({
            "step": "3",
            "role": "主任審査官",
            "content": decision
        })

        return decision

    def run_full_examination(self,
                            dict_a: Dict,
                            dict_b: Dict) -> Dict:
        """
        完全な審査プロセスの実行

        Args:
            dict_a: 本願発明の構造化データ
            app_claims: 本願発明のClaimリスト
            prior_abstract: 先行技術のAbstract
            prior_claims: 先行技術のClaimリスト

        Returns:
            審査結果の辞書
        """
        print("\n" + "🚀" * 40)
        print("特許審査プロセス開始 (統合版)")
        print("🚀" * 40)

        # チャットセッションを開始（文脈保持用）
        self.chat = self.model.start_chat(history=[])

        try:
            # ステップ0: 構造化
            dict_a["step"] = "0.1 Claim"
            dict_b["step"] = "0.2 Candidate Prior Art"
            app_data = self.step0_structure_application(dict_a)
            prior_data = self.step0_structure_application(dict_b)

            # ステップ1: 代理人の主張
            arguments = self.step1_applicant_arguments(app_data, prior_data)

            # ステップ2: 審査官の検証
            review = self.step2_examiner_review(app_data, prior_data, arguments)

            # ステップ3: 最終判断
            decision = self.step3_final_decision(arguments, review)

            print("\n" + "✅" * 40)
            print("特許審査プロセス完了")
            print(decision)
            print("✅" * 40)

            inventiveness = self.judge_inventiveness(decision)

            dodoc_number = dict_b["doc_number"]
            top_k = dict_b["top_k"]

            return {
                "doc_number": dodoc_number,
                "top_k": top_k,
                "application_structure": app_data,
                "prior_art_structure": prior_data,
                "applicant_arguments": arguments,
                "examiner_review": review,
                "final_decision": decision,
                "conversation_history": self.conversation_history,
                "inventiveness": inventiveness
            }

        except Exception as e:
            print(f"\n--- エラーが発生しました ---")
            print(f"エラー内容: {e}")
            # エラー発生時でも部分的な結果を返す
            return {
                "error": str(e),
                "conversation_history": self.conversation_history,
                "partial_results": "処理が途中で中断されました"
            }

    def judge_inventiveness(self, final_decision_text: str) -> Dict[str, bool]:
        """
        最終判断テキストから各クレームの進歩性を抽出
        このjsonテキストを抽出して、json形式で返す。
        ```json
{
  "claim1": {
    "inventive": false,
    "reason": "レイトレーシングにおける処理速度向上ニーズは自明であり、パイプラインの分割・並列化は通常の最適化手段であるため。"
  },
  "claim2": {
    "inventive": false,
    "reason": "Claim 1の並列化が容易想到である場合、各ユニットが異なるレイを処理することは並列処理効率最大化のための技術常識であるため。"
  },
  "claim3": {
    "inventive": false,


        Args:
            final_decision_text: 最終判断のテキスト

        Returns:
            各クレームの進歩性を示す辞書

        """
        inventiveness = {}
        # ’’’json形式の部分を抽出
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', final_decision_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
            try:
                json_data = json.loads(json_text)
                # claimは何番まであるか不明なので、動的に処理
                for claim_key in json_data.keys():
                    if claim_key.startswith("claim"):
                        inventiveness[claim_key] = {
                            'inventive': json_data[claim_key]['inventive'],
                            'reason': json_data[claim_key]['reason']
                        }
                return inventiveness
            except json.JSONDecodeError:
                print("❌ 最終判断のJSONパースに失敗しました。")
                print(final_decision_text)
                return {"error": final_decision_text}



        for claim_num in range(1, 4):
            pattern = rf"### {claim_num}\. Claim {claim_num} .*?\n\*\*判断:\*\* \[(容易想到である|容易想到ではない)\]"
            match = re.search(pattern, final_decision_text, re.DOTALL)
            if match:
                inventiveness[claim_num] = (match.group(1) == "容易想到ではない")
            else:
                inventiveness[claim_num] = None  # 判定できなかった場合

        return inventiveness
    def save_results(self, results: Dict, output_path: str):
        """
        審査結果をJSONファイルに保存

        Args:
            results: 審査結果の辞書
            output_path: 出力ファイルパス
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 結果を保存しました: {output_path}")


# ==================== メイン実行関数 ====================

def llm_entry(doc_dict_a, doc_dict_b):
    """
    エントリポイント: 2つの特許文書から進歩性審査を実行し、結果を返す

    このエントリポイント関数は、本願発明と先行技術の2つの特許文書を比較し、
    段階的な進歩性判断（容易想到性の検証）を実行します。

    Args:
        doc_dict_a (dict): 本願発明のクレーム情報を含む辞書
            必須キー:
                - "abstract" (str): 本願発明のAbstract文
                - "claims" (str): 本願発明のClaims（複数のClaimを含むテキスト）

        doc_dict_b (dict): 先行技術のクレーム情報を含む辞書
            必須キー:
                - "abstract" (str): 先行技術のAbstract文
                - "claims" (str): 先行技術のClaims（複数のClaimを含むテキスト）

    Returns:
        dict: 審査結果の辞書（成功時）、以下の情報を含む:
            - application_structure: 本願発明の構造化データ
            - prior_art_structure: 先行技術の構造化データ
            - applicant_arguments: 代理人の主張テキスト
            - examiner_review: 審査官の検証・反論テキスト
            - final_decision: 主任審査官の最終判断テキスト
            - conversation_history: 処理過程の会話履歴
            - inventiveness: 各クレームの進歩性判断 (claim1, claim2, claim3...)

        None: エラー時

    使用例:
        >>> doc_a = {
        ...     "abstract": "ノズルプレートの機械的頑強性を向上させる...",
        ...     "claims": "Claim 1: ...\nClaim 2: ..."
        ... }
        >>> doc_b = {
        ...     "abstract": "高温環境下での表面特性を改善する...",
        ...     "claims": "Claim 1: ...\nClaim 2: ..."
        ... }
        >>> results = entry(doc_a, doc_b)
        >>> if results:
        ...     print(results['inventiveness'])

    注意事項:
        - 環境変数 GOOGLE_API_KEY が .env ファイルに設定されている必要があります
        - Google Generative AI APIを使用するため、APIキーの取得が必要です
        - 処理には数分かかる場合があります（複数のLLM呼び出しを実行）
    """
    try:

        # .envファイルから環境変数を読み込む
        load_dotenv()

        # APIキーの設定（環境変数から取得）
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ .envファイルにGOOGLE_API_KEYを設定してください")
            return None

        # システムの初期化
        system = PatentExaminationSystemIntegrated(api_key)

        # 完全な審査プロセスの実行
        results = system.run_full_examination(doc_dict_a, doc_dict_b)   

        return results

    except ValueError as e:
        print(f"❌ 初期化エラー: {e}")
        return None
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return None
    
if __name__ == "__main__":
    # ここにテストコードやデバッグコードを記述できます
    pass