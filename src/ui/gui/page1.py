from pathlib import Path
import json
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
import google.generativeai as genai
import re


# --- 既存のインポート ---
from infra.config import PROJECT_ROOT, PathManager, DirNames
from model.patent import Patent
from ui.gui import query_detail
from ui.gui import ai_judge_detail
from ui.gui.search_results_list import search_results_list
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
        if ai_judge_dir.exists():
            json_files = sorted(ai_judge_dir.glob("*.json"))
            if json_files:
                latest_json = json_files[-1]
                with open(latest_json, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                st.session_state.ai_judge_results = results

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
            # アップロードされたファイルの内容が、現在ロード中のものと違う場合のみ処理
            # (Streamlitのリロード対策)
            current_content = st.session_state.get("file_content")

            # まだ読み込んでいない、あるいは内容が変わった場合に実行
            # 注: uploaded_file.getvalue()などで比較する方法もあるが、
            # ここでは簡易的に既存stateの有無で判定し、ボタンなしで即時ロードさせる挙動を維持
            if not current_content:
                 handle_new_upload(uploaded_file)
            else:
                 # すでにロード済みだが、ユーザーが別のファイルをドラッグした場合の検知は
                 # file_uploaderのkeyを変えるか、ID比較が必要だが、今回は簡易実装とする
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

                display_idx = 1
                for idx, result in enumerate(st.session_state.ai_judge_results):
                    # result が None の場合はスキップ
                    if result is None:
                        continue

                    # エラーの場合もスキップ
                    if isinstance(result, dict) and 'error' in result:
                        continue

                    # 紐付き候補の有無を判定
                    claim_rejected = False
                    if 'inventiveness' in result:
                        for claim in result["inventiveness"]:
                            inventiveness = result["inventiveness"][claim]
                            inventive_bool = inventiveness.get('inventive', True)
                            if not inventive_bool:
                                claim_rejected = True
                                break

                    # 公報番号を取得
                    doc_num = result.get('prior_art_doc_number', f"Doc #{display_idx}")

                    # DataFrameの行データを追加
                    df_data.append({
                        '順位': display_idx,
                        '公報番号': doc_num,
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
        for i, doc_num in enumerate(doc_numbers_to_fetch):
            doc_num = str(doc_num)
            year_part = doc_num[:4]
            doc_digit_part = doc_num[4:]
            formatted_doc_number = f"{year_part}-{doc_digit_part}"
            output_doc_number = f"{i + 1} - 特開 {formatted_doc_number}号公報"
            st.write(output_doc_number)
            doc_number_output_number_dict[doc_num] = output_doc_number

        # markdown形式で根拠表示 箇条書きで表示doc_numbers_to_fetchの下に根拠を表示する
        # configでevidence_exstractionディレクトリを取得し、存在チェック
        evidence_extraction_dir = PathManager.get_dir(
            st.session_state.current_doc_number,
            DirNames.EVIDENCE_EXTRACTION
        )

        # ディレクトリ内のファイル存在チェック
        evidence_files = list(evidence_extraction_dir.glob("*.json"))
        if evidence_files:
            st.info(f"📂 参照箇所表示: {len(evidence_files)}件の参照文献が保存されています")

            for doc_num in doc_number_output_number_dict.keys():
                st.markdown(f"### 📑 {doc_number_output_number_dict[doc_num]} の判断根拠")
                # listの要素の文字列にdoc_numが含まれるものを抽出
                for evidence_file in evidence_files:
                    if doc_num in evidence_file.name:
                        with open(evidence_file, "r", encoding="utf-8") as f:
                            evidence_data = json.load(f)
                        for item in evidence_data:
                            verified_evidence_list = item["verified_evidence"]
                            for evidence_dict in verified_evidence_list:
                                for evidence_key in evidence_dict:
                                    evidence_content = evidence_dict[evidence_key]
                                    st.markdown(f"-   {evidence_key}:{evidence_content}")
                                st.divider()



        if st.button("根拠テキスト生成", type="primary"):
            with st.spinner("LLM で根拠箇所を抽出中..."):
                query_patent: Patent = st.session_state.query
                ai_results = st.session_state.ai_judge_results
                run_evidence_extraction_for_doc_numbers(
                    query_patent=query_patent,
                    doc_numbers_to_fetch=doc_numbers_to_fetch,
                    ai_judge_results=ai_results,
                )

            st.success("✅ 根拠テキストを生成し、保存しました。ページ下部に表示されます。")
            st.rerun()
        
        
        evidence_files = list(evidence_extraction_dir.glob("*.json"))
        if evidence_files:
            st.info(f"📂 参照箇所表示: {len(evidence_files)}件の参照文献が保存されています")

            for doc_num in doc_number_output_number_dict.keys():
                st.markdown(f"### 📑 {doc_number_output_number_dict[doc_num]} の判断根拠")

                for evidence_file in evidence_files:
                    if doc_num in evidence_file.name:
                        with open(evidence_file, "r", encoding="utf-8") as f:
                            evidence_data = json.load(f)

                        for item in evidence_data:
                            verified_evidence_list = item["verified_evidence"]
                            for evidence_dict in verified_evidence_list:
                                claim_html = evidence_dict.get("claim_html")
                                prior_html = evidence_dict.get("prior_art_html")
                                reason = evidence_dict.get("reason")

                                if claim_html:
                                    st.markdown("**請求項（根拠箇所を黄色でハイライト）**")
                                    st.markdown(claim_html, unsafe_allow_html=True)

                                if prior_html:
                                    st.markdown("**引用文献（根拠箇所を黄色でハイライト）**")
                                    st.markdown(prior_html, unsafe_allow_html=True)

                                if reason:
                                    st.markdown("**AI審査の理由**")
                                    st.write(reason)

                                st.divider()


        # if "reasons" in st.session_state and st.session_state.reasons:
        #     for i, reason in enumerate(st.session_state.reasons):
        #         st.markdown(f"##### 判断根拠 {i + 1}")
        #         st.code(reason, language="markdown")


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
    """根拠生成用に、AI審査結果から「紐付き候補あり」の doc_number を集めて返すだけにする。

    - AI審査結果(ai_judge_results)から「少なくとも1つの請求項で inventive=False」になった
      先行技術の doc_number を抽出する
    - 類似文献検索結果(df_retrieved)には依存しない
      （全文テキストは BigQuery の get_full_patent_info_by_doc_numbers から取得する）
    """

    rejected_doc_numbers: list[str] = []

    for res in ai_judge_results:
        if not isinstance(res, dict):
            continue
        if res.get("error"):
            continue

        inventiveness = res.get("inventiveness")
        if not isinstance(inventiveness, dict):
            continue

        # 少なくとも1つの請求項で「inventive=False」なら紐付き候補とみなす
        claim_rejected = any(
            isinstance(v, dict) and (v.get("inventive") is False)
            for v in inventiveness.values()
        )
        if not claim_rejected:
            continue

        # 先行技術の番号を取得
        doc_num = res.get("prior_art_doc_number") or res.get("doc_number")
        if not doc_num:
            continue

        doc_num = str(doc_num)
        rejected_doc_numbers.append(doc_num)

    if not rejected_doc_numbers:
        st.info("✅ 紐付き候補がある文献はありませんでした。")
        return []

    # 重複削除しつつ順序維持
    unique_doc_numbers = list(dict.fromkeys(rejected_doc_numbers))
    return unique_doc_numbers


def _highlight_snippets(text: str, snippets: list[str]) -> str:
    """snippets に含まれる部分文字列を <mark> で1回ずつハイライトする"""
    if not text:
        return ""

    highlighted = text
    for s in snippets:
        s = s.strip()
        if not s:
            continue
        # 正規表現でエスケープして、最初の1回だけ置換
        pattern = re.escape(s)
        highlighted = re.sub(
            pattern,
            lambda m: f"<mark>{m.group(0)}</mark>",
            highlighted,
            count=1,
        )
    return highlighted


def _extract_evidence_with_llm(
    claim_text: str,
    prior_art_text: str,
    reason_text: str | None = None,
) -> dict:
    """
    LLM に根拠ペアを作らせ、ハイライト済み HTML を返す

    戻り値:
        {
            "claim_html": "<pre>...</pre>",
            "prior_art_html": "<pre>...</pre>",
            "reason": "・・・"
        }
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 GOOGLE_API_KEY が設定されていません。")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json"},
    )

    base_reason = reason_text or ""
    prompt = f"""
あなたは日本の特許審査官です。
以下の本願請求項テキストと先行技術テキスト、および AI 審査での「発明を否定する理由」に基づき、
「発明を否定する根拠となる請求項の部分」と「対応する先行技術の部分」のペアを抽出してください。

[本願請求項全文]
{claim_text}

[先行技術テキスト全文]
{prior_art_text}

[AI審査での発明否定の理由（参考）]
{base_reason}

出力は必ず JSON 形式のみとし、次の形式にしてください。

{{
  "evidence_pairs": [
    {{
      "claim_snippet": "本願請求項から抜き出した、根拠となる日本語の文またはフレーズ（原文をそのまま）",
      "prior_art_snippet": "先行技術から抜き出した、対応する日本語の文またはフレーズ（原文をそのまま）",
      "explanation": "なぜこのペアが発明を否定する根拠になるのかを簡潔に説明（日本語）"
    }},
    ...
  ]
}}

重要:
- snippet は必ず上記の [本願請求項全文] / [先行技術テキスト全文] からそのまま抜き出してください。
- Markdown や HTML タグ (<mark> 等) は含めないでください。
- 日本語で回答してください。
"""

    response = model.generate_content(prompt)
    try:
        data = json.loads(response.text)
        pairs = data.get("evidence_pairs", [])
    except Exception:
        # LLM が壊れた JSON を返した場合、そもそも JSON としてパースできなかった場合のフォールバック
        pairs = []

    # 抜き出されたテキストを使ってハイライト
    claim_snippets = [p.get("claim_snippet", "") for p in pairs]
    prior_snippets = [p.get("prior_art_snippet", "") for p in pairs]

    claim_html = "<pre>" + _highlight_snippets(claim_text, claim_snippets) + "</pre>"
    prior_html = "<pre>" + _highlight_snippets(prior_art_text, prior_snippets) + "</pre>"

    # 説明はひとまず全部つなげる
    explanations = [p.get("explanation", "").strip() for p in pairs if p.get("explanation")]
    explanation_joined = "\n\n".join(explanations)
    reason_merged = (base_reason + "\n\n" + explanation_joined).strip() if explanation_joined else base_reason

    return {
        "claim_html": claim_html,
        "prior_art_html": prior_html,
        "reason": reason_merged,
    }


def run_evidence_extraction_for_doc_numbers(
    query_patent: Patent,
    doc_numbers_to_fetch: list[str],
    ai_judge_results: list[dict],
):
    """
    - 否定された文献 doc_number ごとに
      * 本願請求項テキスト
      * prior_art テキスト
      * AI審査の理由
      をまとめて LLM に投げる
    - 結果を DirNames.EVIDENCE_EXTRACTION 配下に JSON で保存
    """

    current_doc_number = str(query_patent.publication.doc_number)
    evidence_extraction_dir = PathManager.get_dir(
        current_doc_number,
        DirNames.EVIDENCE_EXTRACTION,
    )
    evidence_extraction_dir.mkdir(parents=True, exist_ok=True)

    # prior_art テキストは generate_reasons() で作った retrieved_docs から取得
    prior_art_text_by_doc: dict[str, str] = {}
    if "retrieved_docs" in st.session_state:
        for d in st.session_state.retrieved_docs:
            prior_art_text_by_doc[str(d["doc_number"])] = d["text"]

    # AI審査結果から「その doc_number で否定された claim の理由」をまとめる
    reason_by_doc: dict[str, str] = {}
    for res in ai_judge_results:
        if not isinstance(res, dict):
            continue
        doc_num = str(res.get("prior_art_doc_number", ""))
        if not doc_num or doc_num not in doc_numbers_to_fetch:
            continue
        inv = res.get("inventiveness", {})
        reasons = []
        for claim_name, v in inv.items():
            if not v.get("inventive", True) and v.get("reason"):
                reasons.append(f"{claim_name}: {v['reason']}")
        if reasons:
            reason_by_doc[doc_num] = "\n\n".join(reasons)

    claim_text = query_patent.claims or ""

    # 各文献について LLM を回して JSON を保存
    for doc_num in doc_numbers_to_fetch:
        prior_text = prior_art_text_by_doc.get(str(doc_num), "")
        reason_text = reason_by_doc.get(str(doc_num), "")

        if not prior_text:
            # prior_art テキストがない場合はスキップ
            continue

        evidence = _extract_evidence_with_llm(
            claim_text=claim_text,
            prior_art_text=prior_text,
            reason_text=reason_text,
        )

        # page1 の既存表示ロジックに合わせて、「verified_evidence」のリストとして保存
        output_obj = [
            {
                "doc_number": str(doc_num),
                "verified_evidence": [
                    evidence  # {"claim_html", "prior_art_html", "reason"}
                ],
            }
        ]

        out_path = evidence_extraction_dir / f"evidence_{doc_num}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    page_1()