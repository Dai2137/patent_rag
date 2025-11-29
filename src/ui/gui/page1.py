from pathlib import Path
import json
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- 既存のインポート ---
from infra.config import PROJECT_ROOT, PathManager, DirNames
from model.patent import Patent
from ui.gui import query_detail
from ui.gui import ai_judge_detail
from ui.gui.prior_art_detail import prior_art_detail
from bigquery.patent_lookup import get_full_patent_info_by_doc_numbers

# 定数
MAX_CHAR = 300
EXCLUDE_DIRS = {
    DirNames.UPLOADED, DirNames.TOPK, "temp", DirNames.QUERY, DirNames.KNOWLEDGE,
    "__pycache__", ".git", ".ipynb_checkpoints"
}

def reset_session_state():
    """セッションステートの初期化"""
    keys_to_reset = [
        "df_retrieved", "matched_chunk_markdowns", "reasons",
        "query", "retrieved_docs", "search_results_df",
        "ai_judge_results", "file_content", "project_dir",
        "current_doc_number", "uploaded_dir"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def load_project_by_id(doc_number: str) -> bool:
    """
    【共通処理】指定された doc_number のプロジェクトデータを読み込み、SessionStateを構築する。
    新規アップロード後も、既存選択時も、最終的にこれを呼ぶことで状態を復元する。
    """
    # 1. ステート初期化
    reset_session_state()

    try:
        # --- A. 基本データ（XML/Query）のロード ---
        uploaded_dir = PathManager.get_uploaded_query_path(doc_number)
        query_file = uploaded_dir / "uploaded_query.txt"

        if not query_file.exists():
            st.error(f"❌ 出願テキストが見つかりません: {query_file}")
            return False

        with open(query_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        # XML解析
        query: Patent = st.session_state.loader.run(query_file)

        # 基本ステート設定
        st.session_state.file_content = file_content
        st.session_state.query = query
        st.session_state.project_dir = uploaded_dir.parent
        st.session_state.uploaded_dir = uploaded_dir
        st.session_state.current_doc_number = doc_number

        # --- B. 検索結果（CSV）のロード (存在すれば) ---
        topk_dir = PathManager.get_topk_results_path(doc_number)
        if topk_dir.exists():
            csv_files = sorted(topk_dir.glob("*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
                search_results_df = pd.read_csv(latest_csv)
                st.session_state.search_results_df = search_results_df
                st.session_state.df_retrieved = search_results_df
                st.session_state.search_results_csv_path = str(latest_csv)

        # --- C. AI審査結果（JSON）のロード (存在すれば) ---
        ai_judge_dir = PathManager.get_ai_judge_result_path(doc_number)
        aj_judge_data_success = False
        if ai_judge_dir.exists():
            json_files = sorted(ai_judge_dir.glob("*.json"))
            if json_files:
                latest_json = json_files[-1]
                with open(latest_json, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                st.session_state.ai_judge_results = results

                if st.session_state.ai_judge_results:
                    aj_judge_data_success = True

        return True

    except Exception as e:
        st.error(f"プロジェクト {doc_number} のロード中にエラーが発生しました: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

def handle_new_upload(uploaded_file: UploadedFile):
    """新規アップロード時の処理：保存してIDを特定し、共通ローダーを呼ぶ"""
    try:
        file_content = uploaded_file.read().decode("utf-8")

        # 1. 一時保存してID解析 (doc_numberを取得するため)
        temp_path = PathManager.get_temp_path("uploaded_query.txt")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        with st.spinner("XMLを解析中..."):
            query: Patent = st.session_state.loader.run(temp_path)
            doc_number = query.publication.doc_number

            if not doc_number:
                st.error("❌ XMLから特許番号(doc_number)が取得できませんでした。")
                return

        # 2. 正規ディレクトリへ移動・保存
        PathManager.move_to_permanent(temp_path, doc_number)

        # 3. 共通ローダーを使ってロード (これで既存フローと合流)
        if load_project_by_id(doc_number):
            st.success(f"✅ 新規プロジェクトを作成・ロードしました: {doc_number}")

    except UnicodeDecodeError:
        st.error("❌ ファイルのエンコーディングが正しくありません。UTF-8形式のファイルをアップロードしてください。")
    except Exception as e:
        st.error(f"❌ アップロード処理に失敗しました: {e}")

def page_1():
    st.title("GENIAC-PRIZE prototype")
    st.subheader("東京大学松尾岩沢研究室コミュニティ")

    mode = st.sidebar.radio("モード選択", ("1. 新規アップロード", "2. 既存文献の表示"))

    # --- 入力エリアの描画 ---
    if mode == "1. 新規アップロード":
        st.header("📝 新規出願の審査")
        uploaded_file = st.file_uploader("1. XML形式の出願をアップロードしてください", type=["xml", "txt"])

        if uploaded_file is not None:
            # アップロードされたファイルの内容を取得
            uploaded_content = uploaded_file.getvalue().decode("utf-8")
            current_content = st.session_state.get("file_content")

            # ファイルの内容が変わった場合、または初回アップロードの場合に処理を実行
            if current_content != uploaded_content:
                handle_new_upload(uploaded_file)
            else:
                # 同じファイルが既にロード済み
                st.info(f"ロード済み: {st.session_state.get('current_doc_number')}")

    else: # 既存文献の表示
        st.header("📂 既存プロジェクトの参照")

        eval_dir = PathManager.EVAL_DIR
        if eval_dir.exists():
            projects = [
                d.name for d in eval_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.') and d.name not in EXCLUDE_DIRS
            ]
            projects.sort(reverse=True)

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_doc = st.selectbox("出願IDを選択してください", projects)
            with col2:
                if st.button("読込", type="primary", width="stretch"):
                    if selected_doc:
                        with st.spinner("ロード中..."):
                            if load_project_by_id(selected_doc):
                                st.success(f"✅ {selected_doc} を読み込みました")
                            # else:
                            #     st.error(f"❌ {selected_doc} のロードに失敗しました")

    # --- 共通メインエリア描画 ---
    # データが正常にロードされている場合のみ表示
    if "query" in st.session_state and st.session_state.get("current_doc_number"):
        st.markdown("---")

        # ドキュメント基本情報
        with st.expander(f"📄 出願データ確認: {st.session_state.current_doc_number}"):
            st.text_area("ファイルの中身", st.session_state.get("file_content", ""), height=150)

        # Step 2以降の共通レンダリング
        render_common_steps()


def render_common_steps():
    """
    Step 2以降の共通処理
    データは既に st.session_state にロードされている前提で動作する
    """

    # --- Step 2: 類似文献検索 ---
    st.header("2. 類似文献の検索")

    has_search_results = 'search_results_df' in st.session_state and st.session_state.search_results_df is not None

    if has_search_results:
        st.info(f"�� 検索結果: {len(st.session_state.search_results_df):,}件 取得済み")

        if st.button("📋 詳細リストを表示", key="goto_search_list"):
            if "検索結果一覧" in st.session_state.page_map:
                st.switch_page(st.session_state.page_map["検索結果一覧"])
            else:
                st.error("ページが見つかりません: 検索結果一覧")
        if st.button("🔄 検索をやり直す", type="primary", key="rerun_search"):
            query_detail.query_detail()
    else:
        st.write("Google Patents Public Dataを用いて類似文献を検索します。")
        if st.button("検索実行", type="primary", key="run_new_search"):
            query_detail.query_detail()

    # --- Step 3: AI審査 ---
    st.header("3. AI審査")

    has_ai_results = 'ai_judge_results' in st.session_state and st.session_state.ai_judge_results

    if has_ai_results:
        # 有効な結果をカウント
        valid_results = [r for r in st.session_state.ai_judge_results if r is not None and not (isinstance(r, dict) and 'error' in r)]

        if len(valid_results) == 0:
            st.warning("⚠️ AI審査の結果がありません。AI審査をやり直してください。")
        else:
            st.info(f"💾 審査結果: {len(valid_results)}件 取得済み")

            with st.expander("審査結果一覧を開く", expanded=True):
                # DataFrameのデータを準備
                df_data = []
                valid_indices = []  # 有効な結果の元のインデックスを保存

                # ai_judge_resultsの存在チェック
                if 'ai_judge_results' not in st.session_state or not st.session_state.ai_judge_results:
                    st.error("❌ AI審査結果が見つかりません。")
                    return

                display_idx = 1
                for idx, result in enumerate(st.session_state.ai_judge_results):
                    # result が None の場合はスキップ
                    if result is None:
                        continue

                    # result が辞書型でない場合はスキップ
                    if not isinstance(result, dict):
                        continue

                    # エラーの場合もスキップ
                    if 'error' in result:
                        continue

                    # 紐付き候補の有無を判定
                    claim_rejected = False
                    if 'inventiveness' in result:
                        try:
                            for claim in result["inventiveness"]:
                                inventiveness = result["inventiveness"][claim]
                                inventive_bool = inventiveness.get('inventive', True)
                                if not inventive_bool:
                                    claim_rejected = True
                                    break
                        except Exception as e:
                            continue

                    # 公報番号を取得
                    try:
                        reference_doc_num = result.get('prior_art_doc_number', f"Doc #{display_idx}")
                    except Exception as e:
                        continue

                    # DataFrameの行データを追加
                    df_data.append({
                        '順位': display_idx,
                        '公報番号': reference_doc_num,
                        '紐付き候補の有無': '有' if claim_rejected else '無'
                    })

                    valid_indices.append(idx)
                    display_idx += 1

                # DataFrameを作成して表示
                if df_data:
                    df = pd.DataFrame(df_data)

                    # 保存用のDataFrameを作成（紐付き候補の有無をTrue/Falseに変換）
                    df_to_save = df.copy()
                    df_to_save['紐付き候補の有無_bool'] = df_to_save['紐付き候補の有無'].map({'有': True, '無': False})

                    # DataFrameを保存
                    doc_number = st.session_state.current_doc_number
                    save_path = PathManager.get_file(doc_number, DirNames.AI_JUDGE_TABLE, "ai_judge_table.csv")
                    df_to_save.to_csv(save_path, index=False, encoding='utf-8-sig')

                    # CSVダウンロードボタン
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 CSV形式でダウンロード",
                        data=csv,
                        file_name='ai_judge_results.csv',
                        mime='text/csv',
                    )

                    # データ行数に応じてスクロール可能なコンテナを使用
                    # 10行を超える場合のみ固定高さでスクロール可能にする
                    use_scrollable = len(df_data) > 10
                    container = st.container(height=450) if use_scrollable else st.container()

                    with container:
                        # ヘッダー行
                        header_cols = st.columns([1, 3, 2, 2])
                        with header_cols[0]:
                            st.markdown("**順位**")
                        with header_cols[1]:
                            st.markdown("**公報番号**")
                        with header_cols[2]:
                            st.markdown("**紐付き候補の有無**")
                        with header_cols[3]:
                            st.markdown("**AI審査の詳細表示**")

                        st.divider()

                        # データ行
                        for i, row_data in enumerate(df_data):
                            idx = valid_indices[i]
                            cols = st.columns([1, 3, 2, 2])

                            with cols[0]:
                                st.write(row_data['順位'])
                            with cols[1]:
                                st.write(row_data['公報番号'])
                            with cols[2]:
                                st.write(row_data['紐付き候補の有無'])
                            with cols[3]:
                                if st.button("詳細", key=f"ai_detail_{idx}", use_container_width=True):
                                    st.session_state.selected_prior_art_idx = idx
                                    if "先行技術詳細" in st.session_state.page_map:
                                        st.switch_page(st.session_state.page_map["先行技術詳細"])
                                    else:
                                        st.error("ページが見つかりません: 先行技術詳細")

        if st.button("🔄 AI審査をやり直す", type="primary", key="rerun_ai_judge"):
             run_ai_judge()
    else:
        st.write("LLMを活用し、新規性・進歩性を審査します。")
        if st.button("AI審査実行", type="primary", key="run_ai_judge_new"):
            if not has_search_results:
                st.warning("⚠️ 先に「2. 類似文献の検索」を実行してください。")
            else:
                run_ai_judge()

    # --- Step 4: 判断根拠出力 ---
    st.header("4. 判断根拠出力")

    if not has_ai_results:
        st.write("⚠️ AI審査を実行すると表示されます。")
    else:
        ai_judge_results = st.session_state.ai_judge_results
        if not ai_judge_results or all(r is None or (isinstance(r, dict) and 'error' in r) for r in ai_judge_results):
            st.warning("⚠️ 有効なAI審査結果がありません。AI審査をやり直してください。")
            return
        
        doc_numbers_to_fetch = generate_reasons(ai_judge_results)
        if doc_numbers_to_fetch is None or len(doc_numbers_to_fetch) == 0:
            return
        current_doc_number = str(st.session_state.current_doc_number)
        year_part = current_doc_number[:4]
        doc_digit_part = current_doc_number[4:]
        formatted_current_doc_number = f"{year_part}-{doc_digit_part}"

        st.write(f"✅特願 {formatted_current_doc_number}に紐づく{len(doc_numbers_to_fetch)}件の文献があります。")

        doc_number_output_number_dict = {}
        for i, reference_doc_num in enumerate(doc_numbers_to_fetch):
            reference_doc_num = str(reference_doc_num)
            year_part = reference_doc_num[:4]
            doc_digit_part = reference_doc_num[4:]
            formatted_doc_number = f"{year_part}-{doc_digit_part}"
            output_doc_number = f"{i + 1} - 特開 {formatted_doc_number}号公報"
            st.write(output_doc_number)
            doc_number_output_number_dict[reference_doc_num] = output_doc_number

        # markdown形式で根拠表示 箇条書きで表示doc_numbers_to_fetchの下に根拠を表示する
        # configでevidence_exstractionディレクトリを取得し、存在チェック
        evidence_extraction_dir = PathManager.get_dir(
            st.session_state.current_doc_number,
            DirNames.EVIDENCE_EXTRACTION
        )

        # ディレクトリ内のファイル存在チェック
        evidence_files = list(evidence_extraction_dir.glob("*.json"))
        if evidence_files:
            st.markdown("## 📂 出願文献の基本情報")
            # stからpatentオブジェクトを取得
            patent = st.session_state.query
            # abstract, claimsを取得し、「概要」、「請求項１」などを結合して長い文字列を作成
            abstract_text = patent.abstract if patent.abstract else "N/A"
            claims_text = "\n".join([f"請求項 {i + 1}: {claim}" for i, claim in enumerate(patent.claims)]) if patent.claims else "N/A"
            long_markdown_text = f"### 概要\n{abstract_text}\n\n### 請求項\n{claims_text}\n"
            st.text_area(
                label="出願の概要と請求項",
                value=long_markdown_text,
                height=300,
                disabled=True # 編集不可（読み取り専用）にする
            )

            st.info(f"📂 参照箇所表示: {len(evidence_files)}件の参照文献が保存されています")
            # doc_numberと表示用の番号の辞書
            for reference_doc_num in doc_number_output_number_dict.keys():
                st.markdown(f"### 📑 {doc_number_output_number_dict[reference_doc_num]} の判断根拠")

                for evidence_file in evidence_files:
                    if reference_doc_num in evidence_file.name:
                        break
                else:
                    st.warning(f"❌ 対応するevidence_extractionファイルが見つかりません: {reference_doc_num}")
                    continue
                display_evidence_section(reference_doc_num, evidence_file)


        if st.button("根拠テキスト生成", type="primary"):
            with st.spinner("BigQueryから特許情報を取得中..."):
                get_full_patent_info_by_doc_numbers(doc_numbers_to_fetch, st.session_state.current_doc_number)

def normalize_text_for_search(text):
    """
    テキストを検索用に正規化（スペース・改行を削除）

    Args:
        text: 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """
    if not text:
        return ""

    # 全角スペース、半角スペース、改行、タブを削除
    normalized = text.replace("　", "").replace(" ", "").replace("\n", "").replace("\t", "")
    return normalized


def parse_paragraph_id_from_quote(source_paragraph_raw, doc_full_content, quote):
    """
    段落IDをパースして、セクション名と段落番号を取得する
    段落IDが不正な形式の場合は、quoteの内容でdoc_full_contentを検索する

    Args:
        source_paragraph_raw: 段落IDの生の文字列（例：'[best_mode_0121]' または '[0168]'）
        doc_full_content: doc_full_contentのJSON辞書
        quote: 引用文（必須）

    Returns:
        tuple: (paragraph_name, paragraph_number) または None（エラーの場合）

    Examples:
        >>> parse_paragraph_id_from_quote("[best_mode_0121]", doc_content, quote)
        ("best_mode", 121)

        >>> parse_paragraph_id_from_quote("[0168]", doc_content, quote_text)
        ("best_mode", 165)  # quoteの内容で検索した結果
    """
    # "[best_mode_0121]" -> "best_mode_0121"
    source_paragraph_id = source_paragraph_raw.strip("[]")

    # "_"が含まれている場合：通常の処理
    if "_" in source_paragraph_id:
        try:
            paragraph_name, paragraph_number_str = source_paragraph_id.rsplit("_", 1)
            paragraph_number = int(paragraph_number_str)
            return (paragraph_name, paragraph_number)
        except (ValueError, AttributeError):
            # パース失敗時は quote で検索にフォールバック
            pass

    # "_"がない場合、または通常のパースに失敗した場合：quoteで検索
    if quote:
        # quoteを正規化
        normalized_quote = normalize_text_for_search(quote)

        if not normalized_quote:
            return None

        # doc_full_contentの各セクションを検索
        section_order = ["technical_field", "background_art", "disclosure", "best_mode"]

        for section_name in section_order:
            section_content = doc_full_content.get("description", {}).get(section_name)

            # disclosureはネストされた辞書の可能性があるため、スキップ
            if isinstance(section_content, dict):
                continue

            if isinstance(section_content, list):
                for paragraph_index, paragraph_text in enumerate(section_content):
                    # 段落テキストを正規化
                    normalized_paragraph = normalize_text_for_search(paragraph_text)

                    # 完全一致または部分一致をチェック
                    if normalized_quote in normalized_paragraph:
                        return (section_name, paragraph_index)

        # 見つからない場合
        return None

    # quoteもない場合
    return None


def display_evidence_section(reference_doc_num, evidence_file):
    """
    証拠ファイルから特定のドキュメント番号に関連する証拠を抽出し、
    明細書の該当箇所をハイライト表示する

    Args:
        reference_doc_num: 参照先行技術文献番号
        evidence_file: 証拠データが格納されたJSONファイルのパス
    """
    paragraph_name_dict = {
        "technical_field": "【技術分野】",
        "background_art": "【背景技術】",
        "disclosure": "【発明の概要】",
        "best_mode": "【発明を実施するための形態】"
    }

    current_doc_number = st.session_state.current_doc_number

    # doc_full_contentファイルの読み込み
    doc_full_content_dir = PathManager.get_dir(current_doc_number, DirNames.DOC_FULL_CONTENT)
    doc_full_content_file = doc_full_content_dir / f"{reference_doc_num}.json"

    if not doc_full_content_file.exists():
        st.warning(f"❌ doc_full_contentファイルが見つかりません: {doc_full_content_file}")
        return

    with open(doc_full_content_file, "r", encoding="utf-8") as f:
        doc_full_content = json.load(f)

    # 証拠データの読み込み
    with open(evidence_file, "r", encoding="utf-8") as f:
        evidence_data_list = json.load(f)

    # --- Step 1: 対象ドキュメントの証拠データを検索 ---
    target_evidence_data = None

    # evidence_data_listが配列の場合
    if isinstance(evidence_data_list, list):
        target_evidence_data = next(
            (item for item in evidence_data_list if item.get("doc_number") == reference_doc_num),
            None
        )
    # 単一オブジェクトの場合
    elif isinstance(evidence_data_list, dict) and evidence_data_list.get("doc_number") == reference_doc_num:
        target_evidence_data = evidence_data_list

    if not target_evidence_data:
        st.info(f"📝 ドキュメント番号 `{reference_doc_num}` に一致する証拠データがありません。")
        return

    # --- Step 2: 全証拠を収集してparagraph_nameでグループ化 ---
    evidence_groups = {}  # {paragraph_name: [{"quote": ..., "explanation": ..., ...}, ...]}

    for item in target_evidence_data.get("evidence_items", []):
        citations = item.get("citations", [])
        claim_scope = item.get("claim_scope", "")

        for citation in citations:
            quote = citation.get("quote", "").strip()
            source_paragraph_raw = citation.get("source_paragraph", "")
            explanation = citation.get("proves", "")

            if not quote or not source_paragraph_raw:
                continue

            # 新しい関数を使って段落IDをパース
            result = parse_paragraph_id_from_quote(source_paragraph_raw, doc_full_content, quote)

            if result is None:
                st.warning(f"⚠️ 段落IDの形式が不正です: `{source_paragraph_raw}` (該当する段落が見つかりません)")
                continue

            paragraph_name, paragraph_number = result
            paragraph_name_japanese = paragraph_name_dict.get(paragraph_name)

            if not paragraph_name_japanese:
                st.warning(f"⚠️ 未対応のセクション: `{paragraph_name}` (段落ID: {source_paragraph_raw})")
                continue

            # グループ化
            if paragraph_name not in evidence_groups:
                evidence_groups[paragraph_name] = []

            evidence_groups[paragraph_name].append({
                "quote": quote,
                "explanation": explanation,
                "paragraph_number": paragraph_number,
                "source_paragraph_id": source_paragraph_raw.strip("[]"),
                "claim_scope": claim_scope
            })

    if not evidence_groups:
        st.info("📝 表示可能な証拠が見つかりませんでした。")
        return

    # --- Step 3: グループごとに証拠詳細と該当箇所を表示 ---
    for paragraph_name, evidence_list in sorted(evidence_groups.items()):
        paragraph_name_japanese = paragraph_name_dict[paragraph_name]

        # doc_full_contentに該当セクションがあるか確認
        if "description" not in doc_full_content or paragraph_name not in doc_full_content["description"]:
            st.warning(f"⚠️ 明細書データ内にセクション `{paragraph_name}` が見つかりません。")
            continue

        paragraph_list = doc_full_content["description"][paragraph_name]

        st.markdown(f"### 📄 {paragraph_name_japanese}")

        # 各証拠番号に対応するquoteをマッピング
        paragraph_quotes = {}  # {paragraph_number: [(quote, claim_scope), ...]}

        for evidence in evidence_list:
            para_num = evidence["paragraph_number"]
            if para_num not in paragraph_quotes:
                paragraph_quotes[para_num] = []
            paragraph_quotes[para_num].append({
                "quote": evidence["quote"],
                "claim_scope": evidence["claim_scope"],
                "explanation": evidence["explanation"]
            })

        # 各証拠の詳細を表示
        for idx, evidence in enumerate(evidence_list, 1):
            with st.expander(f"🔍 証拠 {idx}: {evidence['claim_scope']}", expanded=True):
                st.markdown(f"**一致箇所**")
                st.code(evidence['quote'], language=None)
                st.markdown(f"**一致と判断した理由**  \n{evidence['explanation']}")
                st.markdown(f"**箇所**: 明細書 {paragraph_name_japanese} **段落 {evidence['paragraph_number'] + 1}**")

        st.divider()

        # --- Step 4: 該当セクションの全段落を表示（複数のquoteをハイライト） ---
        display_text_list = []

        for i in range(len(paragraph_list)):
            raw_paragraph = paragraph_list[i]

            # 該当段落の場合：複数のquoteをハイライト処理
            if i in paragraph_quotes:
                clean_paragraph = raw_paragraph.replace("　", "").replace(" ", "")

                # 複数のquoteをすべてハイライト（長い順にソートして部分一致を防ぐ）
                quotes_sorted = sorted(
                    paragraph_quotes[i],
                    key=lambda x: len(x["quote"]),
                    reverse=True
                )

                for quote_info in quotes_sorted:
                    clean_quote = quote_info["quote"].replace("　", "").replace(" ", "")
                    if clean_quote and clean_quote in clean_paragraph:
                        yellow_highlight = f"<mark style='background-color: #ffeb3b;'>{clean_quote}</mark>"
                        clean_paragraph = clean_paragraph.replace(clean_quote, yellow_highlight, 1)

                display_text_list.append(f"<b>【段落 {i+1}】</b> {clean_paragraph}")
            else:
                # 通常の段落
                display_text_list.append(f"【段落 {i+1}】 {raw_paragraph}")

        # 全段落を結合して表示
        if display_text_list:
            full_context_text = "<br><br>".join(display_text_list)
        else:
            full_context_text = "⚠️ 表示可能な段落データがありません。"

        with st.container(height=400):
            st.markdown(
                f"**該当箇所の内容**  \n明細書: {paragraph_name_japanese}  \n\n{full_context_text}",
                unsafe_allow_html=True
            )

        st.divider()

def run_ai_judge():
    """AI審査実行ラッパー"""
    st.session_state.n_topk = len(st.session_state.df_retrieved)
    with st.spinner("審査プロセスを実行中..."):
        results = ai_judge_detail.entry(action="button_click")
        if results:
            st.session_state.ai_judge_results = results
            st.success("✅ AI審査が完了しました。")
            st.rerun()

def generate_reasons(ai_judge_results):
    """根拠生成ロジック"""
    # query_object = st.session_state.query
    # rejected_dfを９件まで表示する
    #９件に満たない場合は、top_kから不足している分を補完する
    competition_rule_max_m = 9
    print(competition_rule_max_m, ": mMaxの設定")


    # eval/{doc_number}/ai_judge_result_tableからcsvを読み込み
    doc_number = st.session_state.current_doc_number
    csv_path = PathManager.get_file(doc_number, DirNames.AI_JUDGE_TABLE, "ai_judge_table.csv")

    if not csv_path.exists():
        st.error(f"❌ AI審査結果テーブルが見つかりません: {csv_path}")
        return

    # CSVを読み込み
    df_ai_judge = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 紐付き候補の有無_boolがTrueのものを抽出
    rejected_df = df_ai_judge[df_ai_judge['紐付き候補の有無_bool'] == True].copy()

    if len(rejected_df) == 0:
        st.info("✅ 紐付き候補がある文献はありませんでした。")
        return

    # rejected_dfにdoc_number列とtop_k列を追加
    rejected_df['doc_number'] = rejected_df['公報番号']
    rejected_df['reject_document_exists'] = rejected_df['紐付き候補の有無_bool']

    actual_limit = min(competition_rule_max_m, len(rejected_df))

    # rejected_dfから全てのdoc_numberを取得
    doc_numbers_to_fetch = rejected_df.head(actual_limit)['doc_number'].tolist()
    reject_document_exists_list = rejected_df.head(actual_limit)['reject_document_exists'].tolist() 
    # reject_document_exists_listがTrueのものだけに絞る
    doc_numbers_to_fetch = [doc_num for doc_num, exists in zip(doc_numbers_to_fetch, reject_document_exists_list) if exists]    
    return doc_numbers_to_fetch

    # # BigQueryから一括で特許情報を取得
    # with st.spinner("BigQueryから特許情報を取得中..."):
    #     get_full_patent_info_by_doc_numbers(doc_numbers_to_fetch, doc_number)


    # # doc_numberをキーとした辞書に変換（高速検索のため）
    # patent_info_dict = {info['doc_number']: info for info in patent_info_list}

    # # retrieved_docsに特許情報を追加または更新
    # if "retrieved_docs" not in st.session_state:
    #     st.session_state.retrieved_docs = []

    # for i, target_row in rejected_df.head(actual_limit).iterrows():
    #     doc_number = target_row['doc_number']

    #     # 対応するretrieved_docsを探す
    #     doc_found = False
    #     for doc in st.session_state.retrieved_docs:
    #         if doc.get('doc_number') == doc_number:
    #             # 既存のdocにBigQueryから取得した情報を追加
    #             if doc_number in patent_info_dict:
    #                 patent_info = patent_info_dict[doc_number]
    #                 doc['title'] = patent_info['title']
    #                 doc['abstract'] = patent_info['abstract']
    #                 doc['claims'] = patent_info['claims']
    #                 doc['description'] = patent_info['description']
    #             doc_found = True
    #             break

    #     # 見つからなかった場合は、新規にdocを作成
    #     if not doc_found and doc_number in patent_info_dict:
    #         patent_info = patent_info_dict[doc_number]
    #         new_doc = {
    #             'doc_number': doc_number,
    #             'title': patent_info['title'],
    #             'abstract': patent_info['abstract'],
    #             'claims': patent_info['claims'],
    #             'description': patent_info['description']
    #         }
    #         st.session_state.retrieved_docs.append(new_doc)

    # st.success(f"✅ {len(patent_info_list)}件の特許情報を取得しました。")




    # st.session_state.reasons = []
    # status_text = st.empty()
    # progress = st.progress(0)
    # final_decision = ai_judge_results[0]["final_decision"] 
    # conversation_history = ai_judge_results[0]["conversation_history"] 
    # inventiveness_keys = dict(ai_judge_results[0]["inventiveness"]).keys()
    # for key in inventiveness_keys:
    #     if key.startswith('claim'):
    #         st.session_state.query.claims.append(key.upper())

    #  # 動作確認用ダミーアクセス
    # (['doc_number', 'top_k', 'application_structure', 'prior_art_structure', 'applicant_arguments', 'examiner_review', 'final_decision', 'conversation_history', 'inventiveness', 'prior_art_doc_number'])

    # for i in range(actual_limit):
    #     status_text.text(f"{i + 1} / {actual_limit} 件目を生成中です...")
    #     if "generator" in st.session_state:
    #         reason = st.session_state.generator.generate(
    #             st.session_state.query,
    #             st.session_state.retrieved_docs[i]
    #         )
    #         st.session_state.reasons.append(reason)
    #     else:
    #         st.error("Generatorが初期化されていません。")
    #         break
    #     progress.progress((i + 1) / actual_limit)

    # status_text.text("生成が完了しました。")

if __name__ == "__main__":
    page_1()