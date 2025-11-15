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

from model.patent import Patent
from ui.gui.utils import create_matched_md, retrieve

# 定数
# TODO: 切り替え可能にする？ 別の場所で管理する？
QUERY_PATH = Path("data/gui/uploaded_query.txt")
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
    st.header("2. 情報探索 + 一致箇所表示")
    step2()

    # 3. 一致箇所表示
    st.header("3. 一致箇所表示")
    step3()

    # 4. 判断根拠出力
    st.header("4. 判断根拠出力")
    step4()

    # その他
    st.subheader("その他")
    step99()


def step1():
    file_content = ""
    uploaded_file: UploadedFile | None = st.file_uploader("1. XML形式の出願を１件アップロードしてください。", type=["xml", "txt"])
    if uploaded_file is not None:
        file_content: str = uploaded_file.read().decode("utf-8")
        st.text_area("ファイルの中身:", file_content, height=200)
        if st.session_state.file_id != uploaded_file.file_id:
            reset_session_state()
            st.session_state.file_id = uploaded_file.file_id
            # ディレクトリが存在しない場合は作成
            QUERY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(QUERY_PATH, "w", encoding="utf-8") as f:
                f.write(file_content)
            st.success("ファイルがアップロードされました。検索結果や画面表示を初期化しました。")


def step2():
    st.write("出願の公開番号（query_id）、出願に紐づく公知例の公開番号（knowledge_id）、公知例の一致箇所（retrieved_chunk）を表示します。")
    if st.button("検索", type="primary"):
        query: Patent = st.session_state.loader.run(QUERY_PATH)
        st.session_state.query = query
        st.session_state.df_retrieved = retrieve(st.session_state.retriever, query)

    if not st.session_state.df_retrieved.empty:
        st.dataframe(st.session_state.df_retrieved[["query_id", "knowledge_id", "retrieved_chunk"]])


def step3():
    st.write(f"一致箇所をハイライトし、その前後{MAX_CHAR}文字まで含めて表示します。")
    n_chunk = len(st.session_state.df_retrieved)
    st.session_state.n_chunk = n_chunk
    if st.button("表示", type="primary"):
        st.session_state.matched_chunk_markdowns = []  # クリア
        for i in range(n_chunk):
            markdown_text = create_matched_md(i, st.session_state.loader, MAX_CHAR)
            st.session_state.matched_chunk_markdowns.append(markdown_text)

    if st.session_state.matched_chunk_markdowns:
        for i, md in enumerate(st.session_state.matched_chunk_markdowns):
            st.markdown(f"##### 一致箇所 {i + 1}/{n_chunk}")
            st.markdown(md, unsafe_allow_html=True)


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
