"""
Patent Evidence Miner (PEM) - High Precision Japanese Edition
特許審査の拒絶理由を裏付ける「決定的な証拠（原文引用）」を抽出・検証するシステム（日本語固定・精度強化版）

【改善点】
1. 堅牢なJSON抽出とエラーハンドリング
2. 文脈を考慮した厳密な検証ロジック
3. Few-shot examples付き高品質プロンプト
4. 再試行メカニズムとフォールバック
5. 特許段落番号の適切な管理
6. 詳細なロギングとデバッグ情報
7. **System Instruction による日本語出力の強制（最重要）**
8. **プロンプト末尾のガードレールによる言語制約の強化**
9. **JSON Mode の明示的使用によるパースエラーの根絶**
10. **テキスト正規化による引用検証精度の向上**
"""

import google.generativeai as genai
import os
import json
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import logging
from enum import Enum
from infra.config import PathManager, DirNames, cfg

# ==========================================
# ロギング設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Data Structures
# ==========================================

class VerificationStatus(Enum):
    """検証ステータス"""
    VERIFIED = "verified"              # 完全一致確認済み
    PARTIAL_MATCH = "partial_match"    # 部分一致（要注意）
    NOT_FOUND = "not_found"            # 原文に存在しない
    CONTEXT_MISMATCH = "context_mismatch"  # 文脈が異なる

@dataclass
class PatentSegment:
    """特許文献の最小テキスト単位（段落または文）"""
    id: str                    # 段落ID（例: "[0025]"）
    text: str                  # テキスト本文
    section: str = "unknown"   # セクション名（background, description等）
    index: int = 0             # セクション内のインデックス

@dataclass
class Citation:
    """個別の引用"""
    quote: str                      # 引用文（原文ママ）
    source_paragraph: str           # 段落番号
    character_count: int            # 文字数
    proves: str = ""                # この引用が証明する内容
    context_before: str = ""        # 前の文
    context_after: str = ""         # 後の文
    is_minimal: bool = True         # 必要最小限か
    is_complete_sentence: bool = True  # 完全な文か

@dataclass
class EvidenceItem:
    """検証済みの証拠アイテム"""
    claim_scope: str                    # 関連する請求項
    assertion: str                      # 審査官の主張（探すべき内容）
    citations: List[Citation]           # 引用のリスト
    verification_status: VerificationStatus  # 検証ステータス
    confidence_score: float = 0.0       # 信頼度スコア（0-1）
    thinking_process: str = ""          # LLMの思考プロセス
    summary: str = ""                   # 拒絶理由書用サマリー

@dataclass
class ExtractionResult:
    """抽出結果の包括的なデータ構造"""
    doc_number: str
    total_assertions: int
    verified_count: int
    partial_count: int
    not_found_count: int
    evidence_items: List[Dict]
    errors: List[str] = field(default_factory=list)

# ==========================================
# 2. Enhanced Prompt Templates
# ==========================================

class EnhancedPrompts:
    """改善版プロンプトテンプレート集（修正済み）"""

    # システム全体への指示
    SYSTEM_INSTRUCTION = """
あなたは日本の特許庁（JPO）の熟練した特許審査官アシスタントです。
あなたの役割は、拒絶理由通知書に記載された主張を裏付ける証拠を、先行技術文献から抽出することです。

【重要不可侵ルール】
1. **言語**: すべての思考プロセス（reasoning/rationale）と出力テキストは、**必ず「日本語」**で記述してください。JSONのキーのみ英語を使用します。
2. **原文尊重**: 特許文献からの引用（quote）は、一字一句変更せず、句読点や空白も含めて原文のまま抽出してください。翻訳や要約は厳禁です。
3. **客観性**: 主張を裏付ける記載がない場合は、正直に「見つからない」と報告してください。ハルシネーション（捏造）は許されません。
"""

    PARSE_ARGUMENTS = """
以下の【審査官の拒絶理由】を分析し、**拒絶の根拠となる本質的な主張**を抽出してください。

# 入力
【審査官の拒絶理由】
{examiner_review}

# 出力要件
1. 拒絶理由の成立に不可欠な技術的主張のみを抽出する。
2. 「新規性がない」「容易に想到できる」といった法的結論は除外する。
3. **rationale（根拠）やassertion（主張）の内容は必ず日本語で記述すること。**

# Output Format (JSON Schema)
```json
{{
  "arguments": [
    {{
      "id": "arg_001",
      "claim_scope": "請求項1 - 構成要件A",
      "assertion": "容器の底部に電池とヒーターを内蔵する構成が開示されている",
      "rationale": "請求項の主要構成要件Aに直接対応するため"
    }}
  ],
  "total_count": 1,
  "confidence": 0.95
}}
```

必ず日本語で出力してください。
"""

    EXTRACT_EVIDENCE = """
あなたは「特許審査官」です。拒絶理由通知書に記載するための証拠を、
先行技術文献から**必要最小限**かつ**正確に**抽出してください。

# Critical Instruction（最重要指示）
拒絶理由通知書では、引用文献の「特に参照すべき箇所」を明確に示す必要があります。
引用は**短く、的確に**、そして**一字一句原文のまま**である必要があります。

# 入力データ
【検証すべき主張 (Assertion)】
{assertion}

【検索対象文献（段落番号付き）】
{full_text}

# 引用抽出の厳格なルール

## ✅ 理想的な引用（Good Practices）
1. **1文のみが理想**
   - 主張を証明できる最も明確な1文を抽出
   - 例: 「本体１０の底部には、リチウムイオン電池２０と、電気ヒーター３０が埋め込まれている。」

2. **必要最小限の長さ**
   - 段落全体ではなく、関連する文のみ
   - 前後の文脈は context として別途記録

3. **構成要件との明確な対応**
   - 引用が主張のどの部分を証明するか明確に

## ❌ 避けるべき引用（Bad Practices）
1. **段落全体の引用**
   - ❌ 段落【0015】全体をコピー
   - ✅ 段落【0015】の関連する1文のみ

2. **複数の技術内容を含む長文**
   - ❌ 「図1を参照すると、...があり、...を有し、...を備え、...である。」（100文字以上）
   - ✅ 主張に直接関連する部分のみを抽出

3. **不要な修飾語を含む**
   - ❌ 「本実施形態の一例として示す飲料容器100は、...」
   - ✅ 「飲料容器100は、...」

## 📋 複数文が必要な場合の処理
主張を証明するために複数の文が必要な場合：
- **それぞれを別の引用として抽出**
- 各引用が独立して意味を持つように
- 各引用に段落番号を付与

例：
```json
{
  "evidence": [
    {
      "quote": "本体１０の底部には、リチウムイオン電池２０が埋め込まれている。",
      "source_id": "[0016]",
      "proves": "電池を内蔵する構成"
    },
    {
      "quote": "本体１０の底部には、電気ヒーター３０が埋め込まれている。",
      "source_id": "[0016]",
      "proves": "ヒーターを内蔵する構成"
    }
  ]
}
```

# Few-shot Examples

## 例1: 理想的な1文抽出

**主張**: 「容器の底部に電池とヒーターを内蔵する構成が開示されている」

**文献抜粋**:
```
[0015] 本実施形態の飲料容器100について、図1を参照して説明する。
[0016] 本体10の底部には、リチウムイオン電池20と、電気ヒーター30が埋め込まれている。
[0017] 制御回路40は、温度センサー50からの信号に基づいて動作する。
```

**❌ 悪い抽出（段落全体）**:
```json
{
  "quote": "本体10の底部には、リチウムイオン電池20と、電気ヒーター30が埋め込まれている。制御回路40は、温度センサー50からの信号に基づいて動作する。",
  "source_id": "[0016]-[0017]"
}
```
問題: 「制御回路」の部分は主張と無関係

**✅ 良い抽出（1文のみ）**:
```json
{
  "found": true,
  "evidence": [
    {
      "quote": "本体10の底部には、リチウムイオン電池20と、電気ヒーター30が埋め込まれている。",
      "source_id": "[0016]",
      "context_before": "本実施形態の飲料容器100について、図1を参照して説明する。",
      "context_after": "制御回路40は、温度センサー50からの信号に基づいて動作する。",
      "reasoning": "段落[0016]の1文が、電池とヒーターを底部に内蔵する構成を明確に開示している",
      "character_count": 45
    }
  ]
}
```

## 例2: 複数文が必要な場合の分割

**主張**: 「温度センサーからの信号に基づいてヒーターを制御する機構が開示されている」

**文献抜粋**:
```
[0020] 制御回路40は、温度センサー50からの温度データを受信する。
[0021] 受信した温度データが設定温度より低い場合、制御回路40は電気ヒーター30への電力供給を増加させる。
```

**❌ 悪い抽出（2文を1つの引用に）**:
```json
{
  "quote": "制御回路40は、温度センサー50からの温度データを受信する。受信した温度データが設定温度より低い場合、制御回路40は電気ヒーター30への電力供給を増加させる。",
  "source_id": "[0020]-[0021]"
}
```
問題: 2つの段落にまたがる長文

**✅ 良い抽出（それぞれを分離）**:
```json
{
  "found": true,
  "evidence": [
    {
      "quote": "制御回路40は、温度センサー50からの温度データを受信する。",
      "source_id": "[0020]",
      "proves": "温度センサーからの信号を受信する構成",
      "character_count": 32
    },
    {
      "quote": "受信した温度データが設定温度より低い場合、制御回路40は電気ヒーター30への電力供給を増加させる。",
      "source_id": "[0021]",
      "proves": "受信した信号に基づいてヒーターを制御する構成",
      "character_count": 50
    }
  ]
}
```

## 例3: 証拠が見つからない場合

**✅ 正直に報告**:
```json
{
  "found": false,
  "reason": "文献中に、主張に対応する具体的な記載が見当たらない。段落[0015]-[0030]を確認したが、該当する構成の開示はない。",
  "searched_sections": ["[0015]", "[0016]", "[0020]", "[0021]", "[0030]"]
}
```

# 出力手順

Step 1: <thinking>タグ内で検索・検証プロセスを記述
```
<thinking>
段落[0015]を確認: 「...」
→ 主張の「〜」という部分に関連するが、「〜」については言及なし。

段落[0016]を確認: 「本体10の底部には、リチウムイオン電池20と、電気ヒーター30が埋め込まれている。」
→ これが主張を直接証明する。
→ この1文のみで十分。前後の文は context として記録。
→ 文字数: 45文字（適切な長さ）

段落[0017]も確認: 制御回路の記載があるが、今回の主張とは無関係。
</thinking>
```

Step 2: JSONブロックで構造化データを出力

# 出力フォーマット

**証拠が見つかった場合**:
```json
{
  "found": true,
  "evidence": [
    {
      "quote": "原文を一字一句そのまま（必要最小限の長さ）",
      "source_id": "[段落番号]",
      "context_before": "引用の直前の文（あれば）",
      "context_after": "引用の直後の文（あれば）",
      "proves": "この引用が証明する具体的内容",
      "reasoning": "なぜこの引用が適切かの説明",
      "character_count": <引用の文字数>
    }
  ],
  "quality_check": {
    "is_minimal": true,
    "is_precise": true,
    "suitable_for_rejection_notice": true
  }
}
```

**証拠が見つからない場合**:
```json
{
  "found": false,
  "reason": "該当する記載が見つからなかった具体的理由",
  "searched_sections": ["確認した段落番号のリスト"]
}
```

# Quality Guidelines（品質ガイドライン）

自己チェック項目:
1. ✅ 引用は60文字以内が理想（最大でも100文字以内）
2. ✅ 各引用は1文のみを含む
3. ✅ 段落全体ではなく、必要な部分のみ
4. ✅ 引用が主張を直接証明する
5. ✅ 原文と完全一致（一字一句変更なし）

**重要**: 拒絶理由通知書での使用を常に意識してください。
審査官が出願人に示す証拠として、明確で読みやすいものである必要があります。

必ず <thinking> と JSON の両方を出力してください。
"""

#     # Step 3: 引用の検証と構造化
#     VERIFY_AND_STRUCTURE = """
# あなたは「拒絶理由通知書作成支援システム」です。
# 抽出された引用が拒絶理由通知書に適した形式かを検証し、構造化してください。

# # タスク
# Step 2で抽出された引用を検証し、拒絶理由通知書での使用に適した形式に構造化する。

# # 入力データ
# 【主張】
# {assertion}

# 【抽出された引用】
# {extracted_evidence}

# 【原文（検証用）】
# {original_segments}

# # 検証項目

# ## 1. 引用の適切性チェック
# - ✅ 引用の長さ: 60文字以内が理想、100文字以内が許容範囲
# - ✅ 文の完全性: 文の途中で切れていないか
# - ✅ 原文一致: 一字一句原文と一致しているか
# - ✅ 最小十分性: 主張を証明するのに必要最小限か

# ## 2. 複数引用の統合可能性チェック
# - 複数の引用が同一段落の連続した文の場合
# - 統合しても60文字以内に収まる場合
# → 統合を検討

# ## 3. 拒絶理由通知書形式への適合性
# 特許庁の記載様式に適合しているか：
# - 段落番号が正確か
# - 引用が「」で囲める形式か
# - 構成要件との対応が明確か

# # 出力フォーマット

# ```json
# {
#   "verification_result": {
#     "all_quotes_valid": true,
#     "issues": [],
#     "warnings": []
#   },
#   "structured_for_rejection_notice": {
#     "claim_element": "構成要件A",
#     "assertion": "容器の底部に電池とヒーターを内蔵する構成",
#     "citations": [
#       {
#         "quote": "本体10の底部には、リチウムイオン電池20と、電気ヒーター30が埋め込まれている。",
#         "source_paragraph": "[0016]",
#         "character_count": 45,
#         "is_minimal": true,
#         "is_complete_sentence": true
#       }
#     ],
#     "summary": "引用文献1の段落[0016]には、「本体10の底部には、リチウムイオン電池20と、電気ヒーター30が埋め込まれている。」と記載されており、本願請求項の構成要件Aが開示されている。"
#   }
# }
# ```

# # 警告・エラーの例

# ## 警告レベル（Warning）
# - 引用が80文字を超える（推奨: 60文字以内）
# - 複数の引用を統合できる可能性がある

# ## エラーレベル（Error）
# - 引用が150文字を超える（拒絶理由書に不適切）
# - 文の途中で切れている
# - 段落番号が不明確

# 出力してください。
# """

# ==========================================
# 3. Utility Functions
# ==========================================

class JSONExtractor:
    """堅牢なJSON抽出ユーティリティ"""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict]:
        """
        複数のフォールバック戦略でJSONを抽出
        """
        strategies = [
            JSONExtractor._extract_from_code_block,
            JSONExtractor._extract_from_brackets,
            JSONExtractor._extract_with_regex,
            JSONExtractor._direct_parse
        ]
        
        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        logger.error("All JSON extraction strategies failed")
        return None
    
    @staticmethod
    def _extract_from_code_block(text: str) -> Optional[Dict]:
        """```json ... ``` ブロックから抽出"""
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    
    @staticmethod
    def _extract_from_brackets(text: str) -> Optional[Dict]:
        """最初の { から最後の } までを抽出"""
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            return json.loads(text[start:end+1])
        return None
    
    @staticmethod
    def _extract_with_regex(text: str) -> Optional[Dict]:
        """正規表現で段階的に抽出"""
        # thinkingタグを除去
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    
    @staticmethod
    def _direct_parse(text: str) -> Optional[Dict]:
        """直接パース（最終手段）"""
        return json.loads(text)

class ThinkingExtractor:
    """思考プロセス抽出ユーティリティ"""
    
    @staticmethod
    def extract_thinking(text: str) -> str:
        """<thinking>タグから思考プロセスを抽出"""
        match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

# ==========================================
# 4. Enhanced Core Logic
# ==========================================

class EnhancedPatentEvidenceMiner:
    """改善版特許証拠マイニングシステム"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Args:
            api_key: Google AI APIキー（Noneの場合は環境変数から自動取得）
            model_name: 使用するGeminiモデル名（Noneの場合はデフォルトモデルを使用）
            max_retries: LLM呼び出しの最大リトライ回数

        Raises:
            ValueError: APIキーが取得できない場合
        """
        # .envファイルから環境変数を読み込む
        load_dotenv()

        # APIキーの取得（引数 > 環境変数の優先順位）
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "API Key is required. "
                "Please set GOOGLE_API_KEY in .env file or pass it as an argument."
            )

        # モデル名の設定（引数 > config > デフォルト値の優先順位）
        if model_name is None:
            model_name = cfg.gemini_llm_name

        genai.configure(api_key=api_key)

        # System Instructionを設定（日本語での出力を強制）
        self.system_instruction = EnhancedPrompts.SYSTEM_INSTRUCTION

        # 通常モデル（system_instruction付き）
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_instruction
        )

        # JSON強制モデル（system_instruction + JSON Mode）
        self.json_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )

        self.max_retries = max_retries
        self.json_extractor = JSONExtractor()
        self.thinking_extractor = ThinkingExtractor()

        logger.info(f"EnhancedPatentEvidenceMiner initialized with model: {model_name} (System Instruction Applied)")
    
    def _call_llm_with_retry(
        self,
        prompt: str,
        use_json_mode: bool = False
    ) -> Optional[str]:
        """リトライ機能付きLLM呼び出し（日本語出力を強制）"""
        model = self.json_model if use_json_mode else self.model

        # 明示的に日本語出力を要求するテキストをプロンプト末尾に追加
        final_prompt = prompt + "\n\n必ず日本語で出力してください (Output in Japanese)."

        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(final_prompt)
                if response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("All retry attempts exhausted")
                    return None
        return None
    
    def _prepare_document_enhanced(
        self, 
        patent_json: Dict
    ) -> Tuple[str, List[PatentSegment]]:
        """
        特許文書の高度な前処理
        - 段落番号の保持
        - セクション情報の保持
        - 階層構造の保持
        """
        segments = []
        full_text_lines = []
        
        desc = patent_json.get("description", {})
        
        for section_name, content in desc.items():
            if isinstance(content, list):
                for idx, text in enumerate(content):
                    if isinstance(text, str) and text.strip():
                        # 特許段落番号を検出または生成
                        para_num = self._extract_paragraph_number(text)
                        if not para_num:
                            para_num = f"[{section_name}_{idx:04d}]"
                        
                        segment = PatentSegment(
                            id=para_num,
                            text=text,
                            section=section_name,
                            index=idx
                        )
                        segments.append(segment)
                        full_text_lines.append(f"{para_num} {text}")
            
            elif isinstance(content, str) and content.strip():
                para_num = f"[{section_name}]"
                segment = PatentSegment(
                    id=para_num,
                    text=content,
                    section=section_name,
                    index=0
                )
                segments.append(segment)
                full_text_lines.append(f"{para_num} {content}")
        
        return "\n".join(full_text_lines), segments
    
    def _extract_paragraph_number(self, text: str) -> Optional[str]:
        """テキストから特許段落番号を抽出"""
        # [0001], 【0001】, ¶0001 などのパターンに対応
        patterns = [
            r'\[(\d{4})\]',
            r'【(\d{4})】',
            r'¶(\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return f"[{match.group(1)}]"
        return None
    
    def _verify_quote_enhanced(
        self, 
        quote: str, 
        all_segments: List[PatentSegment]
    ) -> Tuple[VerificationStatus, str, str, str]:
        """
        改善版引用検証
        Returns: (status, source_id, context_before, context_after)
        """
        # 完全一致チェック（空白の扱いに注意）
        quote_normalized = self._normalize_whitespace(quote)
        
        for i, seg in enumerate(all_segments):
            text_normalized = self._normalize_whitespace(seg.text)
            
            # 完全一致
            if quote_normalized == text_normalized:
                context_before = all_segments[i-1].text if i > 0 else ""
                context_after = all_segments[i+1].text if i < len(all_segments)-1 else ""
                return (
                    VerificationStatus.VERIFIED, 
                    seg.id, 
                    context_before, 
                    context_after
                )
            
            # 部分一致（quoteが文の一部として含まれる）
            if quote_normalized in text_normalized:
                # 文脈が妥当かチェック
                if self._is_context_valid(quote, seg.text):
                    context_before = all_segments[i-1].text if i > 0 else ""
                    context_after = all_segments[i+1].text if i < len(all_segments)-1 else ""
                    return (
                        VerificationStatus.VERIFIED, 
                        seg.id, 
                        context_before, 
                        context_after
                    )
                else:
                    return (
                        VerificationStatus.CONTEXT_MISMATCH,
                        seg.id,
                        "",
                        ""
                    )
        
        # 部分一致も試す（より柔軟に）
        for i, seg in enumerate(all_segments):
            if self._fuzzy_match(quote, seg.text):
                return (
                    VerificationStatus.PARTIAL_MATCH,
                    seg.id,
                    "",
                    ""
                )
        
        return (VerificationStatus.NOT_FOUND, "N/A", "", "")
    
    def _normalize_whitespace(self, text: str) -> str:
        """空白の正規化（意味を保持）"""
        # 連続する空白を1つに
        text = re.sub(r'\s+', ' ', text)
        # 前後の空白を削除
        return text.strip()
    
    def _is_context_valid(self, quote: str, full_text: str) -> bool:
        """引用が文脈的に妥当かチェック"""
        # quoteが文の途中で切れていないかなど
        quote_pos = full_text.find(quote)
        if quote_pos == -1:
            return False
        
        # 引用の前後が文の区切りか確認
        before_char = full_text[quote_pos-1] if quote_pos > 0 else " "
        after_char = full_text[quote_pos+len(quote)] if quote_pos+len(quote) < len(full_text) else " "
        
        # 文の境界記号
        boundaries = {' ', '。', '、', '「', '」', '\n', '\t'}
        
        return before_char in boundaries and after_char in boundaries
    
    def _fuzzy_match(self, quote: str, text: str, threshold: float = 0.85) -> bool:
        """あいまいマッチング（簡易版）"""
        # 正規化後の包含チェック
        quote_chars = set(self._normalize_whitespace(quote).replace(" ", ""))
        text_chars = set(self._normalize_whitespace(text).replace(" ", ""))
        
        if not quote_chars:
            return False
        
        overlap = len(quote_chars & text_chars)
        similarity = overlap / len(quote_chars)
        
        return similarity >= threshold
    
    def run(
        self, 
        review_json: Dict, 
        patent_json: Dict
    ) -> ExtractionResult:
        """メイン実行関数"""
        logger.info("="*70)
        logger.info("🕵️  Enhanced Patent Evidence Mining Started")
        logger.info("="*70)
        
        errors = []
        
        # 1. ドキュメント準備
        try:
            doc_text, segments = self._prepare_document_enhanced(patent_json)
            logger.info(f"📚 Document loaded: {len(segments)} segments")
        except Exception as e:
            logger.error(f"Document preparation failed: {e}")
            return ExtractionResult(
                doc_number="ERROR",
                total_assertions=0,
                verified_count=0,
                partial_count=0,
                not_found_count=0,
                evidence_items=[],
                errors=[f"Document preparation error: {str(e)}"]
            )
        
        # 2. 拒絶理由の解析
        review_text = (
            review_json.get("examiner_review") or 
            review_json.get("final_decision", "")
        )
        
        if not review_text:
            logger.error("No review text found")
            return ExtractionResult(
                doc_number=patent_json.get("doc_number", "Unknown"),
                total_assertions=0,
                verified_count=0,
                partial_count=0,
                not_found_count=0,
                evidence_items=[],
                errors=["No examiner review text provided"]
            )
        
        logger.info("🔍 Step 1: Parsing examiner's arguments...")
        # 安全な文字列置換を使用（{や}のエスケープ問題を回避）
        prompt_1 = EnhancedPrompts.PARSE_ARGUMENTS.replace("{examiner_review}", review_text)

        response_1 = self._call_llm_with_retry(prompt_1, use_json_mode=True)
        if not response_1:
            errors.append("Failed to parse examiner's arguments")
            logger.error("LLM call failed for argument parsing")
            return ExtractionResult(
                doc_number=patent_json.get("doc_number", "Unknown"),
                total_assertions=0,
                verified_count=0,
                partial_count=0,
                not_found_count=0,
                evidence_items=[],
                errors=errors
            )
        
        args_data = self.json_extractor.extract_json_from_text(response_1)
        if not args_data:
            errors.append("Failed to extract JSON from argument parsing")
            logger.error("JSON extraction failed")
            return ExtractionResult(
                doc_number=patent_json.get("doc_number", "Unknown"),
                total_assertions=0,
                verified_count=0,
                partial_count=0,
                not_found_count=0,
                evidence_items=[],
                errors=errors
            )
        
        arguments = args_data.get("arguments", [])
        original_count = len(arguments)
        total_count = args_data.get("total_count", original_count)
        confidence = args_data.get("confidence", 0.0)

        logger.info(f"   ✓ Extracted {original_count} essential assertions (confidence: {confidence:.2f})")

        # 品質チェック（警告のみ、切り詰めはしない）
        if original_count > 8:
            logger.warning(f"   ⚠️  QUALITY WARNING: Unusually high assertion count ({original_count})")
            logger.warning(f"   ⚠️  Expected: 2-5 assertions for typical cases, max 7-8 for complex cases")
            logger.warning(f"   ⚠️  This may indicate the LLM failed to extract only essential claims")
            logger.warning(f"   ⚠️  Review: Check if non-essential technical details were extracted separately")
            errors.append(f"Quality warning: High assertion count ({original_count}) - review for non-essential claims")

        # 低信頼度の警告
        if confidence < 0.7 and confidence > 0:
            logger.warning(f"   ⚠️  LOW CONFIDENCE: LLM confidence is {confidence:.2f}")
            logger.warning(f"   ⚠️  The extracted assertions may need manual review")

        # 分解結果の表示
        if len(arguments) > 0:
            logger.info(f"   📋 Essential claims to verify:")
            for i, arg in enumerate(arguments, 1):
                assertion = arg.get('assertion', '')
                rationale = arg.get('rationale', '')
                logger.info(f"      {i}. {assertion[:100]}...")
                if rationale:
                    logger.debug(f"         Rationale: {rationale[:80]}...")
            logger.info("")  # 空行を追加

        # 3. 各論点の証拠抽出
        verified_items = []
        verified_count = 0
        partial_count = 0
        not_found_count = 0
        for idx, arg in enumerate(arguments, 1):
            assertion = arg.get("assertion", "")
            claim_scope = arg.get("claim_scope", "Unknown")
            
            logger.info(f"\n📌 [{idx}/{len(arguments)}] Assertion: {assertion[:80]}...")

            # 安全な文字列置換を使用（{や}のエスケープ問題を回避）
            prompt_2 = EnhancedPrompts.EXTRACT_EVIDENCE.replace("{assertion}", assertion).replace("{full_text}", doc_text)

            response_2 = self._call_llm_with_retry(prompt_2, use_json_mode=False)
            if not response_2:
                errors.append(f"Failed to extract evidence for: {assertion}")
                not_found_count += 1
                continue
            
            # 思考プロセス抽出
            thinking = self.thinking_extractor.extract_thinking(response_2)
            if thinking:
                logger.debug(f"   💭 Thinking: {thinking[:150]}...")
            
            # JSON抽出
            result = self.json_extractor.extract_json_from_text(response_2)
            if not result:
                errors.append(f"JSON extraction failed for: {assertion}")
                not_found_count += 1
                continue
            
            # 証拠が見つかった場合
            if result.get("found"):
                evidence_list = result.get("evidence", [])

                if not evidence_list:
                    logger.warning(f"   ⚠️  found=true but no evidence provided")
                    not_found_count += 1
                    continue

                # 複数の引用をCitationオブジェクトに変換
                citations = []
                for ev in evidence_list:
                    quote = ev.get("quote", "")
                    source_id = ev.get("source_id", "")
                    char_count = ev.get("character_count", len(quote))

                    # 文字数チェック
                    if char_count > 100:
                        logger.warning(f"   ⚠️  Quote length ({char_count} chars) exceeds recommended limit (100)")

                    # Pythonによる厳密な検証
                    status, true_id, ctx_before, ctx_after = self._verify_quote_enhanced(
                        quote, segments
                    )

                    citation = Citation(
                        quote=quote,
                        source_paragraph=true_id if status == VerificationStatus.VERIFIED else source_id,
                        character_count=char_count,
                        proves=ev.get("proves", ""),
                        context_before=ctx_before,
                        context_after=ctx_after,
                        is_minimal=char_count <= 100,
                        is_complete_sentence=True  # TODO: implement check
                    )
                    citations.append(citation)

                    # ログ出力（各引用ごと）
                    if status == VerificationStatus.VERIFIED:
                        logger.info(f"   ✅ VERIFIED ({char_count}字): {quote[:50]}... (in {true_id})")
                    else:
                        logger.warning(f"   ⚠️  {status.value}: {quote[:50]}...")

                # 全体の検証ステータスを決定
                statuses = [
                    self._verify_quote_enhanced(c.quote, segments)[0]
                    for c in citations
                ]
                if all(s == VerificationStatus.VERIFIED for s in statuses):
                    overall_status = VerificationStatus.VERIFIED
                    verified_count += 1
                elif any(s == VerificationStatus.VERIFIED for s in statuses):
                    overall_status = VerificationStatus.PARTIAL_MATCH
                    partial_count += 1
                else:
                    overall_status = VerificationStatus.NOT_FOUND
                    not_found_count += 1

                item = EvidenceItem(
                    claim_scope=claim_scope,
                    assertion=assertion,
                    citations=citations,
                    verification_status=overall_status,
                    confidence_score=result.get("quality_check", {}).get("confidence", 0.0),
                    thinking_process=thinking
                )

                # dataclassをdictに変換（VerificationStatusもstr化）
                item_dict = asdict(item)
                item_dict['verification_status'] = item.verification_status.value
                verified_items.append(item_dict)
            
            else:
                logger.info(f"   ❌ No evidence found (LLM reported)")
                not_found_count += 1
                verified_items.append({
                    "claim_scope": claim_scope,
                    "assertion": assertion,
                    "found": False,
                    "reason": result.get("reason", "No matching description in document")
                })
        
        # 4. 結果サマリー
        result = ExtractionResult(
            doc_number=patent_json.get("doc_number", "Unknown"),
            total_assertions=len(arguments),
            verified_count=verified_count,
            partial_count=partial_count,
            not_found_count=not_found_count,
            evidence_items=verified_items,
            errors=errors
        )
        
        logger.info("\n" + "="*70)
        logger.info("📊 EXTRACTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Assertions: {result.total_assertions}")
        logger.info(f"✅ Verified: {result.verified_count}")
        logger.info(f"⚠️  Partial/Uncertain: {result.partial_count}")
        logger.info(f"❌ Not Found: {result.not_found_count}")
        if errors:
            logger.warning(f"⚠️  Errors encountered: {len(errors)}")
        
        return result

    def save_result(
        self,
        result: ExtractionResult,
        doc_number: str,
        output_filename: Optional[str] = None
    ) -> None:
        """
        抽出結果をJSONファイルとして保存

        Args:
            result: 抽出結果
            doc_number: 特許公開番号（ディレクトリ識別用）
            output_filename: 出力ファイル名（Noneの場合は自動生成）
        """
        # 出力ディレクトリを取得（evidence_extraction ディレクトリを使用）
        output_dir = PathManager.get_dir(doc_number, DirNames.EVIDENCE_EXTRACTION)

        # ファイル名の生成
        if output_filename is None:
            output_filename = f"evidence_{result.doc_number}.json"

        output_path = output_dir / output_filename

        # 結果を辞書に変換
        output_data = {
            "document": result.doc_number,
            "summary": {
                "total_assertions": result.total_assertions,
                "verified": result.verified_count,
                "partial_match": result.partial_count,
                "not_found": result.not_found_count,
                "verification_rate": f"{(result.verified_count/result.total_assertions*100):.1f}%" if result.total_assertions > 0 else "N/A"
            },
            "evidence_items": result.evidence_items,
            "errors": result.errors if result.errors else None
        }

        # JSONファイルとして保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 結果を保存しました: {output_path}")

# ==========================================
# 5. Entry Point Function
# ==========================================

def llm_entry(review_dict, patent_dict):
    """
    エントリポイント: 審査官の拒絶理由から証拠を抽出し、結果を返す

    このエントリポイント関数は、審査官の拒絶理由と先行技術文献を受け取り、
    拒絶理由を裏付ける証拠を先行技術から抽出・検証します。

    Args:
        review_dict (dict): 審査官の拒絶理由を含む辞書
            必須キー:
                - "examiner_review" (str): 審査官の拒絶理由テキスト
                または
                - "final_decision" (str): 最終決定テキスト

        patent_dict (dict): 先行技術の特許文書を含む辞書
            必須キー:
                - "doc_number" (str): 特許文書番号
                - "description" (dict): 特許の説明セクション

    Returns:
        ExtractionResult: 抽出結果の辞書（成功時）、以下の情報を含む:
            - doc_number: 特許文書番号
            - total_assertions: 検証すべき主張の総数
            - verified_count: 検証済みの証拠数
            - partial_count: 部分一致の証拠数
            - not_found_count: 見つからなかった証拠数
            - evidence_items: 証拠アイテムのリスト
            - errors: エラーメッセージのリスト

        None: エラー時

    使用例:
        >>> review = {
        ...     "examiner_review": "先行技術には、容器の底部に電池と..."
        ... }
        >>> patent = {
        ...     "doc_number": "US-2025-SMART-MUG",
        ...     "description": {...}
        ... }
        >>> result = llm_entry(review, patent)
        >>> if result:
        ...     print(f"検証率: {result.verified_count}/{result.total_assertions}")

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
        miner = EnhancedPatentEvidenceMiner(api_key=api_key)

        # 証拠抽出の実行
        result = miner.run(review_dict, patent_dict)

        return result

    except Exception as e:
        print(f"⚠️ エラーが発生しました: {e}")
        logger.error(f"llm_entry failed: {e}")
        return None


# ==========================================
# 6. Example Usage
# ==========================================

if __name__ == "__main__":
    # 同じダミーデータを使用
    mock_patent = {
        "doc_number": "US-2025-SMART-MUG",
        "description": {
            "summary": "本発明は、液体の温度を自動的に維持する飲料容器に関する。",
            "background": [
                "従来のマグカップは、時間の経過とともに飲料が冷めてしまうという問題があった。",
                "保温性の高い魔法瓶構造も知られているが、能動的な加熱は行わない。"
            ],
            "detailed_description": [
                "図１を参照すると、飲料容器１００は、セラミック製の本体１０を有する。",
                "本体１０の底部には、リチウムイオン電池２０と、電気ヒーター３０が埋め込まれている。",
                "制御回路４０は、液体の温度を監視する温度センサー５０からの信号に基づいて、電気ヒーター３０への電力供給を制御する。",
                "ユーザーは、側面に設けられたタッチパネル６０を操作することで、希望する温度（例えば６０℃）を設定することができる。",
                "さらに、容器１００は、Bluetooth通信モジュール７０を備え、スマートフォンアプリと連携して温度履歴を確認可能である。"
            ]
        }
    }

    mock_review = {
        "examiner_review": """
本願発明（請求項１）の「スマートフォンで温度管理可能な加熱式マグカップ」は、
先行技術文献（US-2025-SMART-MUG）に基づいて容易に想到し得るものである。

詳細な理由は以下の通りである：
1. 先行技術文献には、容器の底部に電池とヒーターを内蔵し、温度センサーに基づいて
   加熱制御を行う構成が開示されている。
2. また、先行技術文献には、Bluetooth等の無線通信手段を用いて外部端末（スマホ等）と
   連携する機能も明記されている。

したがって、本願発明の構成要件はすべて先行技術に記載されている。
        """
    }

    # 実行（APIキーは自動的に.envから読み込まれる）
    try:
        # EnhancedPatentEvidenceMinerを初期化（APIキーは自動取得）
        miner = EnhancedPatentEvidenceMiner()
        result = miner.run(mock_review, mock_patent)

        # 結果を綺麗に出力
        print("\n" + "="*70)
        print("📄 FINAL VERIFICATION REPORT")
        print("="*70)

        output = {
            "document": result.doc_number,
            "summary": {
                "total_assertions": result.total_assertions,
                "verified": result.verified_count,
                "partial_match": result.partial_count,
                "not_found": result.not_found_count,
                "verification_rate": f"{(result.verified_count/result.total_assertions*100):.1f}%" if result.total_assertions > 0 else "N/A"
            },
            "evidence_items": result.evidence_items,
            "errors": result.errors if result.errors else None
        }

        print(json.dumps(output, indent=2, ensure_ascii=False))

        # エラーサマリー
        if result.errors:
            print("\n⚠️  ERRORS ENCOUNTERED:")
            for err in result.errors:
                print(f"   - {err}")

        # 結果をファイルに保存（PathManagerを使用）
        miner.save_result(result, doc_number=result.doc_number)

    except ValueError as e:
        print(f"\n⚠️  初期化エラー: {e}")
        print("Please set GOOGLE_API_KEY in .env file or environment variables")
    except Exception as e:
        print(f"\n⚠️  実行エラー: {e}")
