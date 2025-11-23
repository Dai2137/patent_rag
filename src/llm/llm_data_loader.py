"""
LLM data loader module

This module provides functions to prepare patent data for LLM processing.
"""

import re
import json
import streamlit as st
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from model.patent import Patent
from ui.gui.utils import format_patent_number_for_bigquery
from ui.gui.utils import normalize_patent_id
from bigquery.patent_lookup import find_documents_batch, get_abstract_claims_by_query
from llm.llm_pipeline import llm_entry
from infra.loader.common_loader import CommonLoader
from bigquery.search_path_from_file import search_path
from infra.config import PathManager, DirNames


# 本番では変更
TOP_K = 5  # 上位K件の類似特許を取得
print(f"注意：LLM Data Loader: TOP_K = {TOP_K}")

def entry(action=None):
    if action == "show_page":
        st.write("LLM Data Loader is ready.")
        return

    # 既にpage1のstep2でqueryが読み込まれていればそれを使う
    if "query" in st.session_state and st.session_state.query is not None:
        query = st.session_state.query
    else:
        st.error("⚠️ 先にステップ1でファイルをアップロードしてください。")
        return None

    # doc_numberを取得
    doc_number = query.publication.doc_number
    if not doc_number:
        st.error("❌ 特許番号（doc_number）が取得できませんでした。")
        return None

    save_abstract_claims_query(query, doc_number)
    query_patent_number_a = format_patent_number_for_bigquery(query)
    abstraccts_claims_list = load_patent_b(query_patent_number_a, doc_number)
    results = llm_execution(abstraccts_claims_list, doc_number)
    return results
    
def llm_execution(abstraccts_claims_list, doc_number):
    """LLM実行部分"""
    # q_*.jsonを見つける.pathlibで見つける。glonbを使う
    query_json_dict = read_json("q", doc_number)

    # AI審査結果ディレクトリを取得
    ai_judge_dir = PathManager.get_ai_judge_result_path(doc_number)

    all_results = []
    for i, row_dict in enumerate(abstraccts_claims_list):
        result = llm_entry(query_json_dict, row_dict)

        # 先行技術のdoc_numberを結果に追加
        if result and isinstance(result, dict):
            result['prior_art_doc_number'] = row_dict.get('doc_number', f'先行技術 #{i + 1}')
        # result is Noneの場合もある
        if result is not None:
            all_results.append(result)

        if all_results is None:
            continue
        
        # 結果をJSONファイルとして保存
        json_file_name = f"{i + 1}_{row_dict['doc_number']}.json"
        abs_path = ai_judge_dir / json_file_name
        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    return all_results


def read_json(prefix, doc_number):
    # q_*.jsonを見つける.pathlibで見つける。glonbを使う
    abstract_claims_dir = PathManager.get_dir(doc_number, DirNames.ABSTRACT_CLAIMS)
    json_files = list(abstract_claims_dir.glob(f"{prefix}_*.json"))
    json_file_name = json_files[0] if json_files else None
    # query_json_file_nameを読む
    if not json_file_name:
        print("No JSON file found.")
        return {}
    json_dict = {}
    with open(json_file_name, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    return json_dict

def save_abstract_claims_query(query, doc_number):
    """queryの特許の要約と請求項を取得し、JSONファイルとして保存する"""
    abstract = query.abstract
    claims = query.claims

    output_dict_json = {
        "top_k": "query",
        "doc_number": doc_number,
        "abstract": abstract,
        "claims": claims
    }
    json_file_name = f"q_{doc_number}.json"

    # PathManagerを使用してディレクトリを取得
    abstract_claims_dir = PathManager.get_dir(doc_number, DirNames.ABSTRACT_CLAIMS)
    abs_path = abstract_claims_dir / json_file_name

    with open(abs_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict_json, f, ensure_ascii=False, indent=4)


def load_patent_b(patent_number_a: Patent, doc_number: str):
    """
    patent_number_aに対応するCSVファイルを見つけて、patent_bを読み込む

    Args:
        patent_number_a: Patent Aのオブジェクト
        doc_number: 特許公開番号

    Returns:
        Patent: 読み込んだPatent Bのオブジェクト
    """
    # PathManagerを使用してtopkディレクトリを取得
    topk_dir = PathManager.get_topk_results_path(doc_number)
    csv_files = list(topk_dir.glob("*.csv"))

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
    top_k_df = search_path(df, top_k=TOP_K)

    abstraccts_claims_list =get_abstract_claims_by_query(top_k_df)

    json_file_name = f"top_k_{patent_number_a}.json"

    # PathManagerを使用してabstract_claimsディレクトリを取得
    abstract_claims_dir = PathManager.get_dir(doc_number, DirNames.ABSTRACT_CLAIMS)
    abs_path = abstract_claims_dir / json_file_name

    with open(abs_path, 'w', encoding='utf-8') as f:
        json.dump(abstraccts_claims_list, f, ensure_ascii=False, indent=4)

    return abstraccts_claims_list

def save_abstract_claims_as_json(abstract_claims_list_dict, query_doc_number: str):
    """abstract_claims_list_dictをJSONファイルとして保存する"""
    # PathManagerを使用してabstract_claimsディレクトリを取得
    abstract_claims_dir = PathManager.get_dir(query_doc_number, DirNames.ABSTRACT_CLAIMS)

    for top_k, abstract_claim_dict in enumerate(abstract_claims_list_dict):
        doc_number = abstract_claim_dict[0][0]
        abstract = abstract_claim_dict[0][1]
        claims = abstract_claim_dict[0][2]
        output_dict_json = {
            "top_k": top_k + 1,
            "doc_number": doc_number,
            "abstract": abstract,
            "claims": claims
        }
        json_file_name = f"{top_k + 1}_{doc_number}.json"
        abs_path = abstract_claims_dir / json_file_name

        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict_json, f, ensure_ascii=False, indent=4)
        print(f"Saved abstract and claims to {abs_path}")


def get_abstract_claims(found_lookup):
    # doc_infoのresult_tableで同じresult_tableをまとめる
    result_table_dict = {}  
    for doc_info in found_lookup:
        table_name = doc_info['result_table']
        if table_name not in result_table_dict:
            result_table_dict[table_name] = []
        result_table_dict[table_name].append(doc_info)
    
    abstract_claim_list_dict = get_abstract_claims_by_query(result_table_dict)
    return abstract_claim_list_dict

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
                final_lookup_entrys.append(best_match.to_dict())
    return final_lookup_entrys


if __name__ == "__main__":
    #entry()
    # llm_execution(1)
    load_patent_b('JP-2010000001-A')