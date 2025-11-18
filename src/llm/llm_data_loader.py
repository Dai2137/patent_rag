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
from ui.gui.utils import format_patent_number_for_bigquery
from ui.gui.utils import parse_patent_info
from ui.gui.utils import normalize_patent_id
from bigquery.patent_lookup import find_documents_batch, find_patent_document

# プロジェクトルート（このファイルは src/llm/ にあるので3階層上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# page1と同じQUERY_PATHを使用
QUERY_PATH = PROJECT_ROOT / "eval" / "uploaded" / "uploaded_query.txt"

# query_detailと同じOUTPUT_CSV_PATHを使用
OUTPUT_CSV_PATH = PROJECT_ROOT / "eval" / "topk"

TOP_K = 5  # 上位K件の類似特許を取得

def entry():
    # ステップ１でアップロードしたxmlを読み込む
    
    query = st.session_state.loader.run(QUERY_PATH)
    query_patent_number_a = format_patent_number_for_bigquery(query)
    patent_b: Patent = load_patent_b(query_patent_number_a)
    # print(patent_b)


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
        # pathlib name only stem
        if patent_number_a == str(csv_file.stem):
            csv_file_path = csv_file
            break
    
    if not csv_file_path:
        return None
    
    df = pd.read_csv(csv_file_path)
    # dfの全行をループし、pabulication_numberを取得
    publication_numbers = []
    year_part = []
    counter = 0
    for _, row in df.iterrows():
        if counter >= TOP_K:
            break
        publication_number = row.get('publication_number', None)

        year, doc_number = normalize_patent_id(publication_number)
        if not doc_number:
            continue
        publication_numbers.append(doc_number)
        year_part.append(year)
        counter += 1

    find_document(publication_numbers, year_part)

import re

def find_document(publication_numbers, year_parts):
    target_lookup_entries = find_documents_batch(publication_numbers)
    # target_lookup_entriesを辞書に変換し、dataframeにする
    df_lookup_entries = pd.DataFrame(target_lookup_entries)
    # 下のアルゴリズムをdataframeで実装する
    # doc_number の列に publication_numbersが含まれるものを探す
    # Noneを除外
    publication_numbers = [num for num in publication_numbers if num is not None]

    final_lookup_entrys = []
    for pub_num, year in zip(publication_numbers, year_parts):
        found_df = df_lookup_entries[df_lookup_entries['doc_number'].str.contains(pub_num, na=False)]
        if len(found_df) == 0:
            continue
        if len(found_df) == 1:
            final_lookup_entrys.append(found_df.iloc[0].to_dict())
            continue
        # 複数ヒットした場合、yearでフィルタリング
        if year is not None:
            print(found_df.head())
            found_df_year = found_df[found_df['doc_number'].str.contains(year, na=False)]
            if len(found_df_year) > 0:
                final_lookup_entrys.append(found_df_year.iloc[0].to_dict())
                continue
            else:
                # yearが和暦であれば西暦に変換して再度試す
                imperial = re.match(r'^[HSR]\d{2}$', year)
                if imperial:
                    era = imperial.group()[0]
                    year_num = int(imperial.group()[1:])
                    if era == 'S': year = 1925 + year_num
                    elif era == 'H': year = 1988 + year_num
                    elif era == 'R': year = 2018 + year_num

                # データフレームの doc_number から '年' (先頭4桁) を抽出して整数化
                # エラー処理: 数字でないものが混ざっている場合に備えて coerce を使用
                found_df['extracted_year'] = pd.to_numeric(found_df['doc_number'].str[:4], errors='coerce')
                
                # ターゲットの年 (int化)
                target_year = int(year)
                
                # 「年号の差（絶対値）」を計算
                found_df['year_diff'] = (found_df['extracted_year'] - target_year).abs()
                
                # 年の差が小さい順にソート
                sorted_df = found_df.sort_values('year_diff')
                
                # 最も近い候補を取得
                best_match = sorted_df.iloc[0]
                min_diff = best_match['year_diff']

                # 許容範囲の設定（例: ±3年以内なら採用する）
                YEAR_TOLERANCE = 3
                
                if min_diff <= YEAR_TOLERANCE:
                    print(f"Found closest match: {best_match['doc_number']} (Diff: {min_diff} years)")
                    final_lookup_entrys.append(best_match.to_dict())
                else:
                    # 差が大きすぎる場合は、別の特許の可能性が高いため採用しない（あるいは警告ログを出して採用しない）
                    print(f"Skipping {pub_num}: Closest match {best_match['doc_number']} is {min_diff} years away.")
    return final_lookup_entrys



    # final_lookup_entrys = []
    # for pub_num, year in zip(publication_numbers, year_parts):
    #     # find pub_num from target_lookup_entries
    #     lookup_entrys = []
    #     for entry in target_lookup_entries:
    #         doc_number = entry['doc_number']
    #         if pub_num in doc_number:
    #             if year is None:
    #                 lookup_entrys.append(entry)
    #                 break
    #             else:
    #                 if year in doc_number:
    #                     lookup_entrys.append(entry)
    #                     break
    #     final_lookup_entrys.extend(lookup_entrys)
    # return lookup_entrys
    # target_lookup_entries = []
    # for top_i, number in enumerate(publication_numbers):
    #     if top_i >= TOP_K:
    #         break
    #     entries = find_patent_document(number)
    #     target_lookup_entries.extend(entries)
    # return target_lookup_entries


#     # if csv_file_path is None:
#     #     raise FileNotFoundError(f"patent_number_a '{patent_number_a}' を含むCSVファイルが見つかりません。検索ディレクトリ: {OUTPUT_CSV_PATH}")

#     # # CSVファイルが存在するか確認
#     # if not csv_file_path.exists():
#     #     raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_file_path}")

#     # # CSVファイルを読み込む
#     # df = pd.read_csv(csv_file_path)

#     # # TODO: CSVから最初の類似特許を取得して、そのPatentオブジェクトを返す
#     # # 現時点では実装未完了
#     # raise NotImplementedError("load_patent_b() の完全な実装が必要です")


# def patent_to_dict(patent: Patent) -> Dict[str, Any]:
#     """
#     Convert Patent object to dictionary format.

#     Args:
#         patent: Patent object

#     Returns:
#         Dictionary format patent data
#     """
#     return asdict(patent)

if __name__ == "__main__":
    load_patent_b('JP-2010000001-A')