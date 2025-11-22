"""
必要機能：
- 任意の出願を読み込む機能：XML形式ファイルをアップロードする（中身はXMLだけど、拡張子はtxtとxmlの両方に対応しておいた方がいい）
- 情報探索機能：細かい指定はない　→出願IDと紐づきIDの対応関係を表形式で表示する
- 一致箇所表示機能：細かい指定はない　→テキストを表示し、一致箇所はハイライトさせる
- 判断根拠出力機能：情報探索と一致箇所表示の根拠を自然言語で表示　→判断根拠テキストボックスを作って、その中に「情報探索の根拠」と「一致箇所の根拠」を表示する。
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from infra.config import PathManager
from model.patent import Patent
# from ui.gui.utils import create_matched_md  # , retrieve
from ui.gui import query_detail
from ui.gui import ai_judge_detail

# プロジェクトルート（このファイルは src/ui/gui/ にあるので3階層上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 定数
# TODO: 切り替え可能にする？ 別の場所で管理する？
QUERY_PATH = PROJECT_ROOT / "eval" / "uploaded" / "uploaded_query.txt"
MAX_CHAR = 300


def reset_session_state():
    st.session_state.df_retrieved = pd.DataFrame()
    st.session_state.matched_chunk_markdowns = []
    st.session_state.reasons = []
    st.session_state.query = None
    st.session_state.retrieved_docs = []


def page_1():
    st.title("GENIAC-PRIZE prototype v1.0")
    st.write("1. から 4. までを順番に実行してください。")

    # 1. 任意の出願を読み込む
    st.header("1. 任意の出願を読み込む")
    step1()

    # 2. 情報探索
    st.header("2. 類似文献の検索")
    step2()

    # 3. AI審査
    st.header("3. AI審査")
    step3()

    # 4. 判断根拠出力
    st.header("4. 判断根拠出力")
    step4()

    # その他
    st.subheader("その他")
    step99()


def step1():
    uploaded_file: UploadedFile | None = st.file_uploader("1. XML形式の出願を１件アップロードしてください。", type=["xml", "txt"])
    if uploaded_file is not None:
        # アップロードされたファイルの中身を読み込む
        file_content: str = uploaded_file.read().decode("utf-8")
        st.text_area("ファイルの中身:", file_content, height=200)

        if st.session_state.get("file_content") != file_content:
            # --- Phase 1: 一時ディレクトリに保存 ---
            temp_path = PathManager.get_temp_path("uploaded_query.txt")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            try:
                with st.spinner("XMLを解析中..."):
                    # XMLをparseしてdoc_numberを取得
                    query: Patent = st.session_state.loader.run(temp_path)
                    public_doc_number = query.publication.doc_number

                    if not public_doc_number:
                        st.error("特許番号(doc_number)が取得できませんでした。")
                        return

                # --- Phase 2: 正規のディレクトリにコピー ---
                permanent_path = PathManager.move_to_permanent(temp_path, public_doc_number)

                # アップロードディレクトリのパスを取得
                uploaded_dir = PathManager.get_uploaded_query_path(public_doc_number)

                # Session Stateの更新
                reset_session_state()
                st.session_state.file_content = file_content
                st.session_state.query = query
                st.session_state.project_dir = permanent_path.parent
                st.session_state.source_file = permanent_path
                st.session_state.uploaded_dir = uploaded_dir

                st.success(f"✓ 初期化完了: 特許ID {public_doc_number}")
                st.info(f"📁 データ保存先: {st.session_state.project_dir}")

            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

        else:
            # すでにロード済み
            if "query" in st.session_state and st.session_state.query:
                st.success(f"✓ ロード済み: 特許ID {st.session_state.query.publication.doc_number}")

def step2():
    st.write("出願の公開番号（query_id）について、Google Patents Public Dataの埋め込みベクトルを用いて類似文献を検索し、上位の文献を表示します。")
    st.write("Google Patents Public Dataは、高精度かつ効率のよい埋め込みベクトルを提供しており、特許文献の意味的な類似性を捉えることができます。")
    st.write("このため、独自に膨大な文献のベクトル化が不要となり、コスト的に効率的な検索が可能です。")
    if st.button("検索", type="primary"):
        # query: Patent = st.session_state.loader.run(QUERY_PATH)
        # st.session_state.query = query
        query_detail.query_detail()


def step3():
    st.write(f"一致箇所をハイライトし、その前後{MAX_CHAR}文字まで含めて表示します。")
    n_chunk = len(st.session_state.df_retrieved)
    st.session_state.n_chunk = n_chunk
    if st.button("AI審査", type="primary"):
        ai_judge_detail.ai_judge_detail(action="button_click")

    # if st.session_state.matched_chunk_markdowns:
    #     for i, md in enumerate(st.session_state.matched_chunk_markdowns):
    #         st.markdown(f"##### 一致箇所 {i + 1}/{n_chunk}")
    #         st.markdown(md, unsafe_allow_html=True)


def step4():
    n_chunk = st.session_state.n_chunk

    if st.button("生成", type="primary"):
        st.session_state.reasons = []  # クリア

        status_text = st.empty()
        progress = st.progress(0)

        for i in range(n_chunk):
            status_text.text(f"{i + 1} / {n_chunk} 件目を生成中です...")
            reason = st.session_state.generator.generate(st.session_state.query, st.session_state.retrieved_docs[i])
            st.session_state.reasons.append(reason)
            progress.progress((i + 1) / n_chunk)
        status_text.text("生成が完了しました。")

    if st.session_state.reasons:
        for i, reason in enumerate(st.session_state.reasons):
            st.markdown(f"##### 判断根拠 {i + 1} / {n_chunk}")
            st.code(reason, language="markdown")


def step99():
    st.write("次の出願に対しても同様に、1. から順番に実行してください。")
    if st.button("リセット"):
        reset_session_state()
        st.success("クエリや検索結果の履歴をリセットしました。")
