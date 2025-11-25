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
from ui.gui.search_results_list import search_results_list
from ui.gui.prior_art_detail import prior_art_detail

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
        if st.button("🔄 検索をやり直す", key="rerun_search"):
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
                display_idx = 1
                for idx, result in enumerate(st.session_state.ai_judge_results):

                    # result が None の場合はスキップ（メッセージ表示なし）
                    if result is None:
                        continue

                    # エラーの場合もスキップ（メッセージ表示なし）
                    if isinstance(result, dict) and 'error' in result:
                        continue

                    # 有効なデータのみ表示
                    doc_num = result.get('prior_art_doc_number', f"Doc #{display_idx}")
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(f"**{display_idx}. {doc_num}**")
                    with c2:
                        if st.button("詳細", key=f"ai_detail_{idx}"):
                            st.session_state.selected_prior_art_idx = idx
                            if "先行技術詳細" in st.session_state.page_map:
                                st.switch_page(st.session_state.page_map["先行技術詳細"])
                            else:
                                st.error("ページが見つかりません: 先行技術詳細")

                    display_idx += 1

        if st.button("🔄 AI審査をやり直す", type="primary", key="rerun_ai_judge"):
             run_ai_judge()
    else:
        st.write("LLMを活用し、新規性・進歩性を審査します。")
        if st.button("AI審査実行", type="primary", key="run_ai_judge_new"):
            if not has_search_results:
                st.warning("⚠️ 先に「2. 類似文献の検索」を実行してください。")
            else:
                run_ai_judge()

    ai_judge_results = st.session_state.get("ai_judge_results")
    if ai_judge_results and type(ai_judge_results) is list :

        claim_rejected_results = []
        for ai_result in ai_judge_results:
            # doc_number = ai_result["doc_number"]  
            # final_decision = ai_result["inventiveness"]

            claim_rejected = False 
            for claim in ai_result["inventiveness"]:
                inventiveness = ai_result["inventiveness"][claim]
                inventive_bool = inventiveness['inventive']
                if inventive_bool:
                    continue
                claim_rejected = True
            if claim_rejected:
                claim_rejected_results.append(ai_result)
        if claim_rejected_results:
            st.warning(f"⚠️ 請求項 {len(claim_rejected_results)}件 は進歩性が認められませんでした。")
            
            rejected_dict ={
                'doc_number': [r['doc_number'] for r in claim_rejected_results],
                'top_k': [r['top_k'] for r in claim_rejected_results],
            }
            rejected_df = pd.DataFrame(rejected_dict)
            st.dataframe(rejected_df)
            print()
        else:
            st.success("✅ 全ての請求項で進歩性が認められました。")

    # --- Step 4: 判断根拠出力 ---
    st.header("4. 判断根拠出力")

    if not has_ai_results:
        st.write("⚠️ AI審査を実行すると表示されます。")
    else:
        ai_judge_results = st.session_state.ai_judge_results

        if st.button("根拠テキスト生成", type="primary"):
            # if "retrieved_docs" not in st.session_state or not st.session_state.retrieved_docs:
            #      st.error("文献データ(retrieved_docs)がメモリにありません。再検索が必要な可能性があります。")
            # else:
            generate_reasons(ai_judge_results)

        if "reasons" in st.session_state and st.session_state.reasons:
            for i, reason in enumerate(st.session_state.reasons):
                st.markdown(f"##### 判断根拠 {i + 1}")
                st.code(reason, language="markdown")


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
    st.session_state.reasons = []
    status_text = st.empty()
    progress = st.progress(0)
    final_decision = ai_judge_results[0]["final_decision"] 
    conversation_history = ai_judge_results[0]["conversation_history"] 
    inventiveness_keys = dict(ai_judge_results[0]["inventiveness"]).keys()
    for key in inventiveness_keys:
        if key.startswith('claim'):
            st.session_state.query.claims.append(key.upper())

     # 動作確認用ダミーアクセス
    (['doc_number', 'top_k', 'application_structure', 'prior_art_structure', 'applicant_arguments', 'examiner_review', 'final_decision', 'conversation_history', 'inventiveness', 'prior_art_doc_number'])

    for i in range(actual_limit):
        status_text.text(f"{i + 1} / {actual_limit} 件目を生成中です...")
        if "generator" in st.session_state:
            reason = st.session_state.generator.generate(
                st.session_state.query,
                st.session_state.retrieved_docs[i]
            )
            st.session_state.reasons.append(reason)
        else:
            st.error("Generatorが初期化されていません。")
            break
        progress.progress((i + 1) / actual_limit)

    status_text.text("生成が完了しました。")

if __name__ == "__main__":
    page_1()