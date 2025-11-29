"""
Patent Quote Highlighter (LLM Version)
特許のフルテキストから引用文を特定し、強調表示するシステム（LLM版）

このシステムは、証拠抽出結果（evidence_items）に含まれるquoteを元の特許文献から検索し、
該当箇所を特定して強調表示します。LLMを使用して高精度に位置を特定します。
"""

import google.generativeai as genai
import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
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
# データ構造
# ==========================================

@dataclass
class QuoteLocation:
    """引用箇所の位置情報"""
    quote: str                    # 引用文
    section_name: str             # セクション名（best_mode, background_art等）
    paragraph_index: int          # 段落インデックス（0始まり）
    paragraph_id: str             # 段落ID（例: "[best_mode_0121]"）
    start_char: int               # 段落内の開始文字位置
    end_char: int                 # 段落内の終了文字位置
    found: bool = True            # 見つかったかどうか
    confidence: str = "exact"     # マッチング信頼度（exact, partial, not_found）

# ==========================================
# プロンプトテンプレート
# ==========================================

class QuoteLocatorPrompts:
    """引用箇所特定用プロンプトテンプレート"""

    SYSTEM_INSTRUCTION = """
あなたは特許文献解析の専門家です。
あなたの役割は、証拠として抽出された引用文が、元の特許文献のどこに記載されているかを正確に特定することです。

【重要ルール】
1. **正確性**: 引用文と完全に一致する箇所を探してください。
2. **段落番号**: 実際の段落番号（段落ID）を正確に報告してください。
3. **位置情報**: 段落内での開始位置と終了位置（文字数）を特定してください。
4. **日本語出力**: すべての出力は日本語で記述してください。
"""

    LOCATE_QUOTE = """
以下の引用文が、特許文献のどこに記載されているかを特定してください。

# 引用文
{quote}

# 特許文献（段落番号付き）
{patent_text}

# ヒント情報
{hint_info}

# 出力要件

引用文が見つかった場合は、以下のJSON形式で出力してください：

```json
{{
  "found": true,
  "section_name": "セクション名（例: best_mode, background_art等）",
  "paragraph_id": "段落ID（例: [best_mode_0121]）",
  "paragraph_index": 段落のインデックス番号（0始まり）,
  "paragraph_text": "段落の全文",
  "start_char": 段落内での引用開始位置（文字数）,
  "end_char": 段落内での引用終了位置（文字数）,
  "confidence": "exact",
  "reasoning": "なぜこの箇所だと判断したかの説明"
}}
```

引用文が見つからない場合は、以下のJSON形式で出力してください：

```json
{{
  "found": false,
  "reason": "見つからなかった理由",
  "searched_sections": ["確認したセクション名のリスト"],
  "confidence": "not_found"
}}
```

**重要**:
- 引用文と完全に一致する箇所を探してください（一字一句同じ）
- 段落IDは特許文献に記載されている実際のIDを使用してください
- start_charとend_charは段落テキストの先頭からの文字数（0始まり）で指定してください
- JSONのみを出力し、余計な説明は不要です

必ず日本語で出力してください。
"""

# ==========================================
# LLM引用箇所特定システム
# ==========================================

class LLMQuoteLocator:
    """LLMを使用して引用箇所を特定するクラス"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        初期化

        Args:
            api_key: Google AI APIキー（Noneの場合は環境変数から自動取得）
            model_name: 使用するGeminiモデル名（Noneの場合はデフォルトモデルを使用）
            max_retries: LLM呼び出しの最大リトライ回数
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

        # System Instructionを設定
        self.system_instruction = QuoteLocatorPrompts.SYSTEM_INSTRUCTION

        # JSON強制モデル（system_instruction + JSON Mode）
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )

        self.max_retries = max_retries

        logger.info(f"LLMQuoteLocator initialized with model: {model_name}")

    def _call_llm_with_retry(self, prompt: str) -> Optional[str]:
        """リトライ機能付きLLM呼び出し"""
        final_prompt = prompt + "\n\n必ず日本語で出力してください (Output in Japanese)."

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(final_prompt)
                if response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("All retry attempts exhausted")
                    return None
        return None

    def _prepare_patent_text(self, patent_dict: Dict) -> Tuple[str, Dict[str, List[str]]]:
        """
        特許文献を段落番号付きテキストに変換

        Returns:
            (formatted_text, section_map): フォーマット済みテキストとセクションマップ
        """
        description = patent_dict.get("description", {})
        formatted_lines = []
        section_map = {}

        for section_name, content in description.items():
            if isinstance(content, dict):
                continue

            if isinstance(content, list):
                section_map[section_name] = []
                for idx, paragraph_text in enumerate(content):
                    if isinstance(paragraph_text, str) and paragraph_text.strip():
                        para_id = f"[{section_name}_{idx:04d}]"
                        formatted_lines.append(f"{para_id} {paragraph_text}")
                        section_map[section_name].append(paragraph_text)

            elif isinstance(content, str) and content.strip():
                para_id = f"[{section_name}]"
                formatted_lines.append(f"{para_id} {content}")
                section_map[section_name] = [content]

        return "\n".join(formatted_lines), section_map

    def _create_hint_info(self, source_paragraph: Optional[str]) -> str:
        """ヒント情報を作成"""
        if source_paragraph:
            return f"推定される段落: {source_paragraph}"
        return "ヒント情報なし"

    def locate_quote_in_patent(
        self,
        quote: str,
        patent_dict: Dict,
        source_paragraph_hint: Optional[str] = None
    ) -> QuoteLocation:
        """
        LLMを使用して引用文の位置を特定する

        Args:
            quote: 引用文（一字一句そのまま）
            patent_dict: 特許文献の辞書
            source_paragraph_hint: 段落IDのヒント（例: "[best_mode_0121]"）

        Returns:
            QuoteLocation: 引用箇所の位置情報
        """
        logger.info(f"🔍 LLMで引用箇所を特定中: {quote[:50]}...")

        # 特許文献を準備
        patent_text, section_map = self._prepare_patent_text(patent_dict)
        hint_info = self._create_hint_info(source_paragraph_hint)

        # プロンプト作成
        prompt = QuoteLocatorPrompts.LOCATE_QUOTE.format(
            quote=quote,
            patent_text=patent_text,
            hint_info=hint_info
        )

        # LLM呼び出し
        response = self._call_llm_with_retry(prompt)
        if not response:
            logger.warning(f"❌ LLM呼び出し失敗: {quote[:50]}...")
            return self._create_not_found_location(quote)

        # JSONパース
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response: {response}")
            return self._create_not_found_location(quote)

        # 結果の検証
        if not result.get("found"):
            logger.warning(f"❌ 見つかりませんでした: {quote[:50]}...")
            return self._create_not_found_location(quote)

        # QuoteLocationを作成
        section_name = result.get("section_name", "")
        paragraph_id = result.get("paragraph_id", "")
        paragraph_index = result.get("paragraph_index", -1)
        start_char = result.get("start_char", -1)
        end_char = result.get("end_char", -1)
        confidence = result.get("confidence", "exact")

        # 検証: 実際に引用が存在するか確認
        paragraph_text = result.get("paragraph_text", "")

        # 基本的な検証: 必須フィールドが有効かチェック
        if not paragraph_text or start_char < 0 or end_char <= start_char:
            logger.warning(f"❌ 無効なロケーション情報が返されました")
            logger.debug(f"paragraph_text exists: {bool(paragraph_text)}, start: {start_char}, end: {end_char}")
            return self._create_not_found_location(quote)

        # 引用の一致検証
        extracted_quote = paragraph_text[start_char:end_char]
        if self._normalize_text(extracted_quote) != self._normalize_text(quote):
            logger.warning(f"⚠️ 抽出された引用が元の引用と一致しません")
            logger.debug(f"Expected: {quote}")
            logger.debug(f"Got: {extracted_quote}")
            # 一致しない場合は見つからなかったとして扱う
            return self._create_not_found_location(quote)

        logger.info(f"✅ 見つかりました: {paragraph_id} (confidence: {confidence})")

        return QuoteLocation(
            quote=quote,
            section_name=section_name,
            paragraph_index=paragraph_index,
            paragraph_id=paragraph_id,
            start_char=start_char,
            end_char=end_char,
            found=True,
            confidence=confidence
        )

    def _normalize_text(self, text: str) -> str:
        """テキストを正規化（比較用）"""
        # 空白を統一
        normalized = re.sub(r'\s+', ' ', text)
        return normalized.strip()

    def _create_not_found_location(self, quote: str) -> QuoteLocation:
        """見つからなかった場合のQuoteLocationを作成"""
        return QuoteLocation(
            quote=quote,
            section_name="",
            paragraph_index=-1,
            paragraph_id="",
            start_char=-1,
            end_char=-1,
            found=False,
            confidence="not_found"
        )

# ==========================================
# 強調表示機能
# ==========================================

def highlight_quote_in_paragraph(
    paragraph_text: str,
    quote: str,
    start_char: int,
    end_char: int,
    highlight_format: str = "html"
) -> str:
    """
    段落内の引用箇所を強調表示

    Args:
        paragraph_text: 段落のテキスト
        quote: 引用文
        start_char: 開始位置
        end_char: 終了位置
        highlight_format: 強調形式（"html", "markdown", "ansi", "brackets"）

    Returns:
        強調表示されたテキスト
    """
    before = paragraph_text[:start_char]
    highlighted = paragraph_text[start_char:end_char]
    after = paragraph_text[end_char:]

    if highlight_format == "html":
        return f'{before}<mark style="background-color: yellow; font-weight: bold;">{highlighted}</mark>{after}'
    elif highlight_format == "markdown":
        return f"{before}**{highlighted}**{after}"
    elif highlight_format == "ansi":
        # ANSI color codes
        YELLOW_BG = "\033[43m\033[30m"  # 黄色背景、黒文字
        RESET = "\033[0m"
        return f"{before}{YELLOW_BG}{highlighted}{RESET}{after}"
    else:  # brackets
        return f"{before}【{highlighted}】{after}"

# ==========================================
# メイン処理関数
# ==========================================

def process_evidence_items(
    evidence_data: List[Dict],
    patent_dict: Dict,
    output_format: str = "html",
    api_key: Optional[str] = None
) -> Dict:
    """
    証拠アイテムを処理して引用箇所を強調表示

    Args:
        evidence_data: 証拠抽出結果のリスト（evidence_items）
        patent_dict: 特許文献（2021536169.jsonの形式）
        output_format: 出力形式（"html", "markdown", "ansi", "brackets"）
        api_key: Google AI APIキー（Noneの場合は環境変数から取得）

    Returns:
        強調表示された結果を含む辞書
    """
    locator = LLMQuoteLocator(api_key=api_key)
    results = []

    total_quotes = 0
    found_quotes = 0

    # 各証拠アイテムを処理
    for evidence_item in evidence_data:
        if not isinstance(evidence_item, dict):
            continue

        evidence_result = {
            "claim_scope": evidence_item.get("claim_scope", ""),
            "assertion": evidence_item.get("assertion", ""),
            "citations": []
        }

        # 各引用を処理
        citations = evidence_item.get("citations", [])
        for citation in citations:
            quote = citation.get("quote", "")
            source_paragraph = citation.get("source_paragraph", "")
            proves = citation.get("proves", "")

            if not quote:
                continue

            total_quotes += 1

            # LLMで引用箇所を特定
            location = locator.locate_quote_in_patent(
                quote=quote,
                patent_dict=patent_dict,
                source_paragraph_hint=source_paragraph
            )

            if location.found:
                found_quotes += 1

            citation_result = {
                "quote": quote,
                "proves": proves,
                "source_paragraph": source_paragraph,
                "location": {
                    "found": location.found,
                    "section_name": location.section_name,
                    "paragraph_index": location.paragraph_index,
                    "paragraph_id": location.paragraph_id,
                    "confidence": location.confidence
                }
            }

            # 引用箇所が見つかった場合、強調表示を生成
            if location.found:
                section_content = patent_dict.get("description", {}).get(location.section_name, [])
                if isinstance(section_content, list) and location.paragraph_index < len(section_content):
                    paragraph_text = section_content[location.paragraph_index]

                    highlighted_text = highlight_quote_in_paragraph(
                        paragraph_text=paragraph_text,
                        quote=quote,
                        start_char=location.start_char,
                        end_char=location.end_char,
                        highlight_format=output_format
                    )

                    citation_result["highlighted_paragraph"] = highlighted_text
                    citation_result["original_paragraph"] = paragraph_text
                    citation_result["paragraph_number"] = location.paragraph_index + 1

            evidence_result["citations"].append(citation_result)

        # foundがFalseの証拠アイテムも含める
        if "found" in evidence_item and not evidence_item["found"]:
            evidence_result["found"] = False
            evidence_result["reason"] = evidence_item.get("reason", "")

        results.append(evidence_result)

    return {
        "doc_number": patent_dict.get("doc_number", ""),
        "invention_title": patent_dict.get("invention_title", ""),
        "evidence_count": len(results),
        "total_quotes": total_quotes,
        "found_quotes": found_quotes,
        "not_found_quotes": total_quotes - found_quotes,
        "success_rate": f"{(found_quotes/total_quotes*100):.1f}%" if total_quotes > 0 else "N/A",
        "highlighted_evidence": results
    }

# ==========================================
# HTML出力生成関数
# ==========================================

def generate_html_output(result: Dict, output_path: str):
    """
    強調表示されたHTMLレポートを生成

    Args:
        result: process_evidence_itemsの出力結果
        output_path: 出力HTMLファイルパス
    """
    html_parts = []

    # HTMLヘッダー
    html_parts.append("""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>特許引用箇所の強調表示（LLM版）</title>
    <style>
        body {
            font-family: 'Yu Gothic', 'Meiryo', sans-serif;
            line-height: 1.8;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .stats {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats-item {
            display: inline-block;
            margin-right: 30px;
        }
        .evidence-item {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .claim-scope {
            font-size: 1.2em;
            font-weight: bold;
            color: #2980b9;
            margin-bottom: 10px;
        }
        .assertion {
            background-color: #fff9e6;
            border-left: 4px solid #f39c12;
            padding: 10px;
            margin: 10px 0;
        }
        .citation {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .quote-box {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 10px 0;
            font-style: italic;
        }
        .location-info {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 5px 0;
        }
        .highlighted-paragraph {
            background-color: #ffffff;
            border: 2px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            line-height: 2;
        }
        mark {
            background-color: yellow;
            font-weight: bold;
            padding: 2px 4px;
        }
        .proves {
            color: #16a085;
            margin: 10px 0;
            font-weight: 500;
        }
        .not-found {
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 5px;
        }
        .separator {
            border-top: 2px dashed #bdc3c7;
            margin: 30px 0;
        }
        .llm-badge {
            background-color: #9b59b6;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }
    </style>
</head>
<body>
""")

    # ヘッダー
    html_parts.append(f"""
    <div class="header">
        <h1>📄 特許引用箇所の強調表示レポート<span class="llm-badge">🤖 LLM版</span></h1>
        <p>特許番号: {result['doc_number']}</p>
        <p>発明の名称: {result['invention_title']}</p>
    </div>
    """)

    # 統計情報
    html_parts.append(f"""
    <div class="stats">
        <div class="stats-item"><strong>証拠数:</strong> {result['evidence_count']}</div>
        <div class="stats-item"><strong>総引用数:</strong> {result['total_quotes']}</div>
        <div class="stats-item"><strong>発見数:</strong> {result['found_quotes']}</div>
        <div class="stats-item"><strong>未発見数:</strong> {result['not_found_quotes']}</div>
        <div class="stats-item"><strong>成功率:</strong> {result['success_rate']}</div>
    </div>
    """)

    # 警告メッセージの表示（証拠がない場合）
    if 'warning' in result:
        html_parts.append(f"""
    <div class="not-found" style="margin: 20px 0; padding: 20px; font-size: 1.1em;">
        <h3>⚠️ 警告</h3>
        <p>{result['warning']}</p>
        <p style="margin-top: 15px;"><strong>対処方法:</strong></p>
        <ul>
            <li>証拠抽出プロセスを実行してください</li>
            <li>または、evidence_extraction/{result['doc_number']}.json ファイルの内容を確認してください</li>
        </ul>
    </div>
    """)

    # 各証拠アイテム
    for idx, evidence in enumerate(result['highlighted_evidence'], 1):
        html_parts.append(f'<div class="evidence-item">')
        html_parts.append(f'<div class="claim-scope">🔍 証拠 {idx}: {evidence["claim_scope"]}</div>')
        html_parts.append(f'<div class="assertion"><strong>主張:</strong> {evidence["assertion"]}</div>')

        # foundがFalseの場合
        if "found" in evidence and not evidence["found"]:
            html_parts.append(f'<div class="not-found"><strong>⚠️ 該当箇所なし:</strong> {evidence.get("reason", "")}</div>')
        else:
            # 引用を表示
            for cit_idx, citation in enumerate(evidence.get("citations", []), 1):
                html_parts.append(f'<div class="citation">')
                html_parts.append(f'<h3>引用 {cit_idx}</h3>')

                html_parts.append(f'<div class="quote-box">"{citation["quote"]}"</div>')

                if citation.get("proves"):
                    html_parts.append(f'<div class="proves"><strong>証明内容:</strong> {citation["proves"]}</div>')

                location = citation.get("location", {})
                if location.get("found"):
                    confidence = location.get("confidence", "exact")
                    html_parts.append(f'<div class="location-info">📍 位置: {location.get("paragraph_id", "")} （{location.get("section_name", "")} セクション, 段落 {citation.get("paragraph_number", "")}） [信頼度: {confidence}]</div>')

                    if "highlighted_paragraph" in citation:
                        html_parts.append('<h4>📌 元の段落（強調表示）:</h4>')
                        html_parts.append(f'<div class="highlighted-paragraph">{citation["highlighted_paragraph"]}</div>')
                else:
                    html_parts.append(f'<div class="not-found">❌ 該当箇所が見つかりませんでした</div>')

                html_parts.append('</div>')  # citation

        html_parts.append('</div>')  # evidence-item

        if idx < result['evidence_count']:
            html_parts.append('<div class="separator"></div>')

    # HTMLフッター
    html_parts.append("""
</body>
</html>
""")

    # ファイルに書き込み
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    logger.info(f"📄 HTMLレポートを生成しました: {output_path}")

# ==========================================
# エントリーポイント
# ==========================================

def highlight_quotes_entry(
    evidence_json_path: str,
    patent_json_path: str,
    output_json_path: Optional[str] = None,
    output_html_path: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    証拠JSONと特許JSONから引用箇所を強調表示（LLM版）

    Args:
        evidence_json_path: 証拠データのJSONファイルパス（2023120212.json形式）
        patent_json_path: 特許データのJSONファイルパス（2021536169.json形式）
        output_json_path: JSON出力ファイルパス（Noneの場合はスキップ）
        output_html_path: HTML出力ファイルパス（Noneの場合はスキップ）
        api_key: Google AI APIキー（Noneの場合は環境変数から取得）

    Returns:
        処理結果の辞書
    """
    # JSONファイルを読み込み
    with open(evidence_json_path, 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)

    with open(patent_json_path, 'r', encoding='utf-8') as f:
        patent_dict = json.load(f)

    # 特許文献のdoc_numberを取得
    target_doc_number = patent_dict.get("doc_number", "")

    # 証拠データが配列の場合、target_doc_numberと一致する要素を検索
    evidence_items = []
    if isinstance(evidence_data, list):
        # 配列内から該当するdoc_numberの証拠を検索
        for item in evidence_data:
            if isinstance(item, dict) and item.get("doc_number") == target_doc_number:
                evidence_items = item.get("evidence_items", [])
                logger.info(f"✓ 対象文献 {target_doc_number} の証拠を発見しました")
                break
        else:
            # 見つからない場合は警告を出力
            logger.warning(f"⚠️ evidence_dataに doc_number={target_doc_number} の証拠が見つかりません")
            logger.warning(f"   利用可能なdoc_numbers: {[item.get('doc_number') for item in evidence_data if isinstance(item, dict)]}")
            # 空のリストを返すことで、エラーではなく「証拠なし」として処理
    else:
        # 辞書形式の場合は従来通り
        evidence_items = evidence_data.get("evidence_items", [])

    # 処理実行
    logger.info("="*70)
    logger.info("引用箇所の強調表示を開始（LLM版）")
    logger.info("="*70)

    # 証拠がない場合は、空の結果を生成
    if len(evidence_items) == 0:
        logger.warning(f"⚠️ 証拠データが空のため、空の結果を生成します")
        result = {
            "doc_number": patent_dict.get("doc_number", ""),
            "invention_title": patent_dict.get("invention_title", ""),
            "evidence_count": 0,
            "total_quotes": 0,
            "found_quotes": 0,
            "not_found_quotes": 0,
            "success_rate": "N/A",
            "highlighted_evidence": [],
            "warning": f"この文献（{target_doc_number}）の証拠データが見つかりませんでした。証拠抽出プロセスを実行してください。"
        }
    else:
        result = process_evidence_items(
            evidence_data=evidence_items,
            patent_dict=patent_dict,
            output_format="html",
            api_key=api_key
        )

    # JSON結果を保存
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 JSON結果を保存しました: {output_json_path}")

    # HTMLレポートを生成
    if output_html_path:
        generate_html_output(result, output_html_path)

    logger.info("\n" + "="*70)
    logger.info("処理完了（LLM版）")
    logger.info("="*70)
    logger.info(f"証拠数: {result['evidence_count']}")
    logger.info(f"総引用数: {result['total_quotes']}")
    logger.info(f"発見数: {result['found_quotes']} ({result['success_rate']})")
    logger.info(f"未発見数: {result['not_found_quotes']}")

    return result

# ==========================================
# 標準パスを使用したエントリーポイント
# ==========================================

def generate_highlighted_html_for_reference(
    reference_doc_num: str,
    current_doc_number: str
) -> Dict:
    """
    参照文献番号から自動的にパスを取得してHTMLを生成
    PathManagerを使用して標準的なディレクトリ構造からパスを取得する

    Args:
        reference_doc_num: 参照先行技術文献番号
        current_doc_number: 現在審査中の申請特許の番号

    Returns:
        処理結果の辞書

    Raises:
        FileNotFoundError: 必要なファイルが見つからない場合
    """
    # PathManagerから標準的なパスを取得
    evidence_json_path = PathManager.get_dir(
        current_doc_number,
        DirNames.EVIDENCE_EXTRACTION
    ) / f"{reference_doc_num}.json"

    patent_json_path = PathManager.get_dir(
        current_doc_number,
        DirNames.DOC_FULL_CONTENT
    ) / f"{reference_doc_num}.json"

    output_html_path = PathManager.get_dir(
        current_doc_number,
        DirNames.HIGHLIGHTED_EVIDENCE
    ) / f"{reference_doc_num}_highlighted.html"

    output_json_path = PathManager.get_dir(
        current_doc_number,
        DirNames.HIGHLIGHTED_JSON
    ) / f"{reference_doc_num}.json"

    # ファイルの存在確認
    if not evidence_json_path.exists():
        raise FileNotFoundError(f"証拠ファイルが見つかりません: {evidence_json_path}")

    if not patent_json_path.exists():
        raise FileNotFoundError(f"特許文献ファイルが見つかりません: {patent_json_path}")

    # 既存の関数を呼び出し
    logger.info(f"📄 {reference_doc_num} の強調表示HTML & JSONを生成中...")
    result = highlight_quotes_entry(
        evidence_json_path=str(evidence_json_path),
        patent_json_path=str(patent_json_path),
        output_json_path=str(output_json_path),
        output_html_path=str(output_html_path)
    )

    logger.info(f"✅ {reference_doc_num} の強調表示HTMLを生成しました: {output_html_path}")
    logger.info(f"✅ {reference_doc_num} のJSONを保存しました: {output_json_path}")
    return result

# ==========================================
# 使用例
# ==========================================

if __name__ == "__main__":
    # サンプル使用例
    import sys

    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python 'Highlight patent quotes.py' <証拠JSONパス> <特許JSONパス> [出力JSONパス] [出力HTMLパス]")
        print("\n例:")
        print("  python 'Highlight patent quotes.py' evidence.json patent.json output.json output.html")
        sys.exit(1)

    evidence_file = sys.argv[1]
    patent_file = sys.argv[2]
    output_json = sys.argv[3] if len(sys.argv) > 3 else None
    output_html = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        result = highlight_quotes_entry(
            evidence_json_path=evidence_file,
            patent_json_path=patent_file,
            output_json_path=output_json,
            output_html_path=output_html
        )

        print("\n✅ 処理が完了しました")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)