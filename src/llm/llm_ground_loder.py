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
from llm.patent_evidence_miner_improved import llm_entry
from infra.loader.common_loader import CommonLoader
from bigquery.search_path_from_file import search_path
from infra.config import PathManager, DirNames
from llm.llm_ground_passage import evidence_extraction_entry
from llm.patent_evidence_miner_improved import EnhancedPatentEvidenceMiner

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
        json_file_name = f"{row_dict['top_k']}_{row_dict['doc_number']}.json"
        abs_path = ai_judge_dir / json_file_name
        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    return all_results

def load_patent_b(doc_number: str):
    """
    AI審査結果と特許文献の完全な内容から証拠を抽出する

    Args:
        doc_number: 特許公開番号

    Returns:
        list: 証拠抽出結果のリスト
    """

    # ai_judge_result_dirを取得
    ai_judge_result_dir = PathManager.get_ai_judge_result_path(doc_number)
    ai_judge_json_files = list(ai_judge_result_dir.glob("*.json"))

    # AI審査結果ファイルを辞書に格納（doc_number -> ファイルパスのマッピング）
    ai_judge_file_dict = {}
    for judge_json_file in ai_judge_json_files:
        topk, ai_judge_doc_number = judge_json_file.stem.split("_", 1)
        ai_judge_file_dict[ai_judge_doc_number] = judge_json_file

    # 特許文献の完全な内容が格納されているディレクトリを取得
    full_content_dir = PathManager.get_dir(doc_number, DirNames.DOC_FULL_CONTENT)
    json_files = list(full_content_dir.glob("*.json"))

    # 証拠抽出結果を格納するリスト
    extraction_results = []

    # EnhancedPatentEvidenceMinerのインスタンスを作成
    try:
        miner = EnhancedPatentEvidenceMiner()
        print("✅ EnhancedPatentEvidenceMiner初期化成功")
    except ValueError as e:
        print(f"❌ EnhancedPatentEvidenceMiner初期化エラー: {e}")
        print("   従来のevidence_extraction_entryを使用します")
        miner = None

    for json_file in json_files:
        # 対応するAI審査結果ファイルを取得
        evidence_file_name = json_file.stem
        reason_file_path = ai_judge_file_dict.get(evidence_file_name, None)
        if not reason_file_path:
            continue

        try:
            # AI審査結果（拒絶理由など）を読み込む
            with open(reason_file_path, 'r', encoding='utf-8') as f:
                reason_json = json.load(f)

            # 特許文献の完全な内容を読み込む
            with open(json_file, 'r', encoding='utf-8') as f:
                json_contents = json.load(f)

            # データが空の場合はスキップ
            if not reason_json or not json_contents:
                continue

            # 証拠抽出を実行（EnhancedPatentEvidenceMinerを使用）
            if miner is not None:
                # 新しいEnhancedPatentEvidenceMinerを使用
                evidence_result_list = [] 
                for reason_json_dict in reason_json:
                    extraction_result_obj = miner.run(reason_json_dict, json_contents)
                    # ExtractionResultオブジェクトを辞書形式に変換
                    evidence_result = asdict(extraction_result_obj)
                    evidence_result_list.append(evidence_result)

                    if evidence_result:
                        # EnhancedPatentEvidenceMinerの結果の場合は'errors'フィールドをチェック
                        if 'errors' in evidence_result:
                            # エラーリストが空または存在しない場合は成功
                            if not evidence_result.get('errors'):
                                extraction_results.append(evidence_result)
                            else:
                                print(f"⚠️ 証拠抽出エラー: {json_file.stem}")
                                print(f"   エラー内容: {evidence_result.get('errors')}")
                        # 従来のevidence_extraction_entryの結果の場合は'error'フィールドをチェック
                        elif 'error' not in evidence_result:
                            extraction_results.append(evidence_result)
                        else:
                            print(f"⚠️ 証拠抽出エラー: {json_file.stem}")
                            print(f"   エラー内容: {evidence_result.get('error', 'Unknown error')}")

                        # extraction_resultをeval/{doc_number}/evidence_extraction/に保存
                        evidence_extraction_dir = PathManager.get_dir(doc_number, DirNames.EVIDENCE_EXTRACTION)
                        evidence_json_file_full_name = f"{evidence_file_name}.json"
                        evidence_json_path = evidence_extraction_dir / evidence_json_file_full_name
                        with open(evidence_json_path, 'w', encoding='utf-8') as f:
                            json.dump(extraction_results, f, ensure_ascii=False, indent=4)  

        except Exception as e:
            print(f"❌ ファイル処理中にエラーが発生しました: {json_file.name}")
            print(f"   エラー内容: {e}")
            continue
    
    return extraction_results


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



def convert_fullcontent_bigquery_result_to_json(doc_number: str):
    """
    BigQueryの結果JSONファイルを読み込み、個別のドキュメントJSONファイルとして保存する

    Args:
        doc_number: 特許公開番号

    Returns:
        None

    Raises:
        FileNotFoundError: 指定されたディレクトリまたはファイルが見つからない場合
        json.JSONDecodeError: JSONファイルのパースに失敗した場合
        OSError: ファイルの読み書きに失敗した場合
    """
    try:
        # PathManagerを使用してtopkディレクトリを取得
        topk_dir = PathManager.get_himotuki_doc_contents(doc_number)

        if not topk_dir.exists():
            raise FileNotFoundError(f"Directory not found: {topk_dir}")

        json_files = list(topk_dir.glob("*.json"))

        if not json_files:
            print(f"Warning: No JSON files found in {topk_dir}")
            return

        json_content_list = []
        for json_file in json_files:
            try:
                # jsonファイルを開いて、publication.numberを読む
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_content_list = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON file {json_file}: {e}")
                continue
            except OSError as e:
                print(f"Error: Failed to read file {json_file}: {e}")
                continue

            if not isinstance(json_content_list, list):
                print(f"Warning: Expected list in {json_file}, got {type(json_content_list)}")
                continue

            for json_content in json_content_list:
                if not isinstance(json_content, dict):
                    print(f"Warning: Skipping non-dict item in {json_file}")
                    continue

                file_name_doc_number = json_content.get('doc_number', None)

                if not file_name_doc_number:
                    print(f"Warning: 'doc_number' not found in content from {json_file}, skipping")
                    continue

                try:
                    # doc_nuberをファイル名として、eval/{doc_number}/doc_full_content/に保存
                    # ディレクトリ管理はPathManagerに任せる
                    doc_full_content_dir = PathManager.get_dir(doc_number, DirNames.DOC_FULL_CONTENT)

                    # JSONファイルとして保存
                    json_file_name = f"{file_name_doc_number}.json"
                    abs_path = doc_full_content_dir / json_file_name

                    with open(abs_path, 'w', encoding='utf-8') as f:
                        json.dump(json_content, f, ensure_ascii=False, indent=4)

                    print(f"Saved full document content to {abs_path}")
                except OSError as e:
                    print(f"Error: Failed to write file {abs_path}: {e}")
                    continue
                except Exception as e:
                    print(f"Error: Unexpected error while processing doc_number={file_name_doc_number}: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error: Unexpected error in convert_fullcontent_bigquery_result_to_json: {e}")
        raise


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
    load_patent_b('2023104947')