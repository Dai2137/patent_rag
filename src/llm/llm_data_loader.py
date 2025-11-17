"""
LLM data loader module

This module provides functions to prepare patent data for LLM processing.
"""
import streamlit as st
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from model.patent import Patent
from infra.loader.common_loader import CommonLoader
from ui.gui.utils import format_patent_number_for_bigquery

# プロジェクトルート（このファイルは src/llm/ にあるので3階層上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# page1と同じQUERY_PATHを使用
QUERY_PATH = PROJECT_ROOT / "eval" / "uploaded" / "uploaded_query.txt"

# query_detailと同じOUTPUT_CSV_PATHを使用
OUTPUT_CSV_PATH = PROJECT_ROOT / "eval" / "topk"


def entry():
    # ステップ１でアップロードしたxmlを読み込む
    
    query = st.session_state.loader.run(QUERY_PATH)
    query_patent_number_a = format_patent_number_for_bigquery(query)
    patent_b: Patent = load_patent_b(query_patent_number_a)
    # print(patent_b)

    # ステップ１の方法でxmlからpatent情報を取得
    # abstractとclaimsを得る。
    # dict_templateをコピーしてdict_aとする
    # 上で取得したabstract, claimsを設定

    # dict_template = {"abstract": "",
    #                  "claims": []}
    
    # result = {
    #     'doc_a': doc_a_dict,
    #     'doc_b': doc_b_dict,
    #     'comparison': {
    #         'doc_a_publication': doc_a_dict.get('publication', {}).get('doc_number', 'N/A'),
    #         'doc_b_publication': doc_b_dict.get('publication', {}).get('doc_number', 'N/A'),
    #         'doc_a_title': doc_a_dict.get('invention_title', 'N/A'),
    #         'doc_b_title': doc_b_dict.get('invention_title', 'N/A'),
    #     }
    # }

    # return result


def load_patent_b(patent_number_a: Patent) -> Patent:
    """
    patent_number_aに対応するCSVファイルを見つけて、patent_bを読み込む

    Args:
        patent_number_a: Patent Aのオブジェクト

    Returns:
        Patent: 読み込んだPatent Bのオブジェクト
    """
    # OUTPUT_CSV_PATHこの中の*.csvを全部取得
    csv_files = list(OUTPUT_CSV_PATH.glob("*.csv"))

    # # 取得したパス名にpatent_number_aが含まれているものを見つける
    csv_file_path = None
    for csv_file in csv_files:
        if patent_number_a == str(csv_file.name):
            csv_file_path = csv_file
            break
    
    if not csv_file:
        return None
    
    df = pd.read_csv(csv_file_path)
    # dfの全行をループし、pabulication_numberを取得
    publication_numbers = []
    for index, row in df.iterrows():
        publication_number = row.get('publication_number', None)
        if publication_number:
            publication_numbers.append(publication_number)


    # if csv_file_path is None:
    #     raise FileNotFoundError(f"patent_number_a '{patent_number_a}' を含むCSVファイルが見つかりません。検索ディレクトリ: {OUTPUT_CSV_PATH}")

    # # CSVファイルが存在するか確認
    # if not csv_file_path.exists():
    #     raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_file_path}")

    # # CSVファイルを読み込む
    # df = pd.read_csv(csv_file_path)

    # # TODO: CSVから最初の類似特許を取得して、そのPatentオブジェクトを返す
    # # 現時点では実装未完了
    # raise NotImplementedError("load_patent_b() の完全な実装が必要です")


def patent_to_dict(patent: Patent) -> Dict[str, Any]:
    """
    Convert Patent object to dictionary format.

    Args:
        patent: Patent object

    Returns:
        Dictionary format patent data
    """
    return asdict(patent)
