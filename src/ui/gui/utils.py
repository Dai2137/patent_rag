import re
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.documents import Document

# from app.retriever import Retriever
from infra.loader.common_loader import CommonLoader
from model.patent import Patent


# TODO: 検索実行はGUIではなくRetrieverやRAG側で制御すべきか考える。
# def retrieve(retriever: Retriever, query: Patent) -> pd.DataFrame:
#     """
#     検索を実行して、検索結果を返す
#     """
#     query_ids: list[str] = []
#     knowledge_ids: list[str] = []
#     retrieved_paths: list[str] = []
#     retrieved_chunks: list[str] = []

#     retrieved_docs: list[Document] = retriever.retrieve(query)
#     st.session_state.retrieved_docs = retrieved_docs

#     for doc in retrieved_docs:
#         query_ids.append(query.publication.doc_number)
#         knowledge_ids.append(doc.metadata["publication_number"])
#         retrieved_paths.append(doc.metadata["path"])
#         retrieved_chunks.append(doc.page_content)

#     df = pd.DataFrame(
#         {
#             "query_id": query_ids,
#             "knowledge_id": knowledge_ids,
#             "retrieved_path": retrieved_paths,
#             "retrieved_chunk": retrieved_chunks,
#         }
#     )
#     return df


def _normalize_text(text: str) -> str:
    """
    改行・タブ・半角/全角スペースなどの空白文字を全て除去して返す
    """
    if text is None:
        return ""
    # \s で英数字系空白、\u3000 で全角スペース
    return re.sub(r"[\s\u3000]+", "", text)


def create_matched_md(index: int, xml_loader: CommonLoader, MAX_CHAR: int) -> str:
    """
    一致箇所とその前後MAX_CHAR文字を含めMarkdownテキストを作成する。
    一致箇所をハイライト表示するためにHTMLタグを追加する。
    """
    chunk: str = st.session_state.df_retrieved["retrieved_chunk"].iloc[index]
    path: str = st.session_state.df_retrieved["retrieved_path"].iloc[index]

    knowledge: Patent = xml_loader.run(Path(path))
    knowledge_str: str = knowledge.to_str()

    normalized_chunk = _normalize_text(chunk)
    normalized_knowledge = _normalize_text(knowledge_str)

    parts: list[str] = normalized_knowledge.split(normalized_chunk)
    first_part: str = parts[0]
    second_part: str = parts[1] if len(parts) > 1 else ""

    markdown_text = f"""
        {first_part[-MAX_CHAR:]}
        <span style="background-color: yellow; color: black; padding: 2px 4px; border-radius: 3px;">{normalized_chunk}</span>
        {second_part[:MAX_CHAR]}
        """
    return markdown_text

from google.cloud import bigquery
from dotenv import load_dotenv
import os
# .envファイルから環境変数を読み込む
load_dotenv()

# ----------------------------------------------------
# ▼▼▼ ユーザー設定 ▼▼▼
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")


def format_patent_number_for_bigquery(patent: Patent) -> str:
    """
    PatentオブジェクトからBigQuery用の特許番号フォーマット（JP-XXXXX-X）を生成する。

    Args:
        patent: Patentオブジェクト

    Returns:
        BigQuery用にフォーマットされた特許番号（例: JP-2012173419-A, JP-7550342-B2）
    """
    doc_number = patent.publication.doc_number
    country = patent.publication.country or "JP"
    kind = patent.publication.kind


    # BigQueryクライアントの初期化
    client = bigquery.Client(project=PROJECT_ID)


    # SQLクエリ(LIKE演算子で部分一致)
    query = f"""
    SELECT publication_number
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE publication_number LIKE '%{doc_number}%'
    """

    # クエリの実行
    query_job = client.query(query)

    # 結果の取得
    results = query_job.result()

    # 結果の表示
    for row in results:
        formatted_number =  str(row.publication_number)
        
        return formatted_number
    print("該当する特許番号が見つかりませんでした。")
    return ""


def normalize_patent_id(patent_id):
    """
    特許IDを解析し、DB検索用の固定長フォーマット（西暦4桁 + 0埋め6桁）に変換する。
    
    戻り値:
        変換成功時 -> "2010058462" のような文字列
        対象外/不可 -> None
    """
    parts = patent_id.split('-')
    
    # フォーマットチェック
    if len(parts) < 3:
        return (None, None)

    raw_number = parts[1]  # 真ん中 (例: H076, 2011005843)
    kind_code = parts[2]   # 末尾 (例: A, B2)

    # 1. まず種別コードでフィルタリング（特許A, Bのみ対象）
    if not re.match(r'^[AB]', kind_code):
        return (None, None)

    # 変換結果を格納する変数
    year_part = None   # 西暦4桁 (int)
    number_part = None # 番号部分 (str)

    # --- パターン解析 ---

    # パターンA: WO (国際公開再公表) -> WO2014030240
    match_wo = re.match(r'^WO(\d{4})(\d+)$', raw_number)
    if match_wo:
        year_part = int(match_wo.group(1))
        number_part = match_wo.group(2)

    # パターンB: 和暦 (H076, S606174)
    elif re.match(r'^([HSMTR])(\d{2})(\d+)$', raw_number):
        match_imp = re.match(r'^([HSMTR])(\d{2})(\d+)$', raw_number)
        era = match_imp.group(1)
        year_part = era + match_imp.group(2)
        number_part = match_imp.group(3)
        
        # 西暦変換
        # if era == 'S': year_part = 1925 + year_num
        # elif era == 'H': year_part = 1988 + year_num
        # elif era == 'R': year_part = 2018 + year_num

    # パターンC: 西暦4桁付き (2011005843)
    elif re.match(r'^(19|20)\d{2}(\d+)$', raw_number) and len(raw_number) >= 10:
        year_part = raw_number[:4]
        number_part = raw_number[4:]

    # パターンD: 年号なし登録番号 (5021568)
    # これは「年号4桁+6桁」のルールには当てはまらないため、そのままの番号を使う
    elif re.match(r'^\d+$', raw_number):
        return_number = raw_number.zfill(6)
        return (year_part, return_number)

    # --- 整形処理 (年号がある場合) ---
    if year_part is not None and number_part is not None:
        # 番号部分を6桁にゼロ埋めする (例: "6" -> "000006")
        padded_number = number_part.zfill(6)
        # 結合して返す
        return (year_part, padded_number)
    
    return (None, None)

# # --- テスト実行 ---
# ids = [
# #    "JP-2010058462-A",    # ユーザー指定フォーマットの基準
#     "JP-H076-A",          # 問題の短い番号: H07(1995) + 6
# #   "JP-WO2014030240-A1", # WO + 2014 + 030240
#     "JP-S606174-Y2",      # 実用新案 -> 除外されるべき(None)
# #    "JP-5021568-B2",      # 登録番号 -> そのまま返す
#     "JP-H084831-A"        # H08(1996) + 4831
# ]

# print(f"{'INPUT ID':<20} | {'SEARCH KEY (Normalized)'}")
# print("-" * 50)

# search_keys = []
# for pid in ids:
#     key = normalize_patent_id(pid)
#     print(f"{pid:<20} | {str(key)}")
#     if key:
#         search_keys.append(key)




def parse_patent_info(patent_id):
    """
    特許IDを解析し、コア番号の抽出と、特許(A/B)かどうかの判定を行う
    """
    # 1. ハイフンで分割
    parts = patent_id.split('-')
    
    # 想定外のフォーマット（JP-xxxx-xx の形式でない場合）
    if len(parts) < 3:
        return {
            "original": patent_id,
            "core_number": None,
            "kind_code": None,
            "is_target_patent": False,
            "note": "Format Error"
        }
    
    raw_number = parts[1]  # 真ん中 (例: H084831)
    kind_code = parts[2]   # 末尾 (例: Y2)
    
    # 2. コア番号の抽出（ご提示のロジックを活用）
    core_number = raw_number
    
    # パターンA: 和暦付き (例: H084831, S606174) -> 年号除去
    match_imperial = re.match(r'^([HSMTR]\d{2})(\d+)$', raw_number)
    if match_imperial:
        core_number = match_imperial.group(2)
        
    # パターンB: 西暦4桁付き (例: 2011005843) -> 年号除去
    # 条件: 19xx or 20xx で始まり、かつ全体が10桁以上
    elif re.match(r'^(19|20)\d{2}(\d+)$', raw_number) and len(raw_number) >= 10:
        core_number = raw_number[4:]

    # 3. 種別コードによるフィルタリング（ここを追加）
    # 今回の要件: 特許(A, B系)のみ対象。実用新案(Y, U)などは除外。
    # Kind Codeが "A" または "B" で始まるものをTrueとする
    is_target = False
    if re.match(r'^[AB]', kind_code):
        is_target = True

    return {
        "original": patent_id,
        "core_number": core_number,
        "kind_code": kind_code,
        "is_target_patent": is_target
    }

# # --- テスト実行 ---

# ids = [
#     "JP-5021568-B2",    # [対象] 年号なし登録特許
#     "JP-H084831-Y2",    # [除外] 和暦付き実用新案 (Y2) -> 番号抽出はするが対象外
#     "JP-2011005843-A",  # [対象] 西暦付き公開特許
#     "JP-S606174-Y2",    # [除外] 和暦付き実用新案 (昭和)
#     "JP-2023123456-A",  # [対象] 最近の公開特許
#     "JP-INVALID-FMT"    # [エラー] フォーマット不正
# ]

# print(f"{'ORIGINAL ID':<20} | {'CORE':<10} | {'KIND':<5} | {'TARGET?'}")
# print("-" * 55)

# valid_entries = []

# for pid in ids:
#     info = parse_patent_info(pid)
    
#     # 表示用処理
#     core = info['core_number'] if info['core_number'] else "N/A"
#     kind = info['kind_code'] if info['kind_code'] else "N/A"
#     is_target = "YES" if info['is_target_patent'] else "NO"
    
#     print(f"{pid:<20} | {core:<10} | {kind:<5} | {is_target}")

#     # 実際に検索に使うリストを作成する場合
#     if info['is_target_patent']:
#         valid_entries.append(info)



 
def format_patent_number_for_bigquery_compose_id(patent: Patent) -> str:
    """
    PatentオブジェクトからBigQuery用の特許番号フォーマット（JP-XXXXX-X）を生成する。

    Args:
        patent: Patentオブジェクト

    Returns:
        BigQuery用にフォーマットされた特許番号（例: JP-2012173419-A, JP-7550342-B2）
    """
    doc_number = patent.publication.doc_number
    country = patent.publication.country or "JP"
    kind = patent.publication.kind

    # kindから種別コード（A, B, B2など）を抽出
    kind_code = ""
    if kind:
        # 日本語のkind（例: "公開特許公報(A)", "特許公報(B2)"）からコードを抽出
        import re
        match = re.search(r'\(([AB]\d?)\)', kind)
        if match:
            kind_code = match.group(1)

    # kindが取得できない場合は、デフォルトでAを使用
    if not kind_code:
        kind_code = "A"

    # フォーマット: JP-{doc_number}-{kind_code}
    formatted_number = f"{country}-{doc_number}-{kind_code}"

    return formatted_number
