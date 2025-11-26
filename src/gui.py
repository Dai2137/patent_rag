import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# .envファイルを読み込み
from infra.config import PROJECT_ROOT
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from app.generator import Generator
# from app.retriever import Retriever
from infra.config import cfg
from infra.loader.common_loader import CommonLoader
from ui.gui.page1 import page_1
from ui.gui.query_detail import query_detail
from ui.gui.ai_judge_detail import ai_judge_detail
from ui.gui.prior_art_detail import prior_art_detail
from ui.gui.search_results_list import search_results_list

# 定数
# TODO: GUI関連の定数の適切な定義場所を考える。移動する。
# KNOWLEDGE_DIR は infra.config.PathManager.KNOWLEDGE_DIR を使用


# セッションステート
# TODO: データはRepositoryクラス、処理はRAGクラスなどにラップしたい。
def init_session_state():
    # 不変
    if "loader" not in st.session_state:
        st.session_state.loader = CommonLoader()
    # if "retriever" not in st.session_state:
    #     st.session_state.retriever = Retriever(knowledge_dir=KNOWLEDGE_DIR)
    if "generator" not in st.session_state:
        st.session_state.generator = Generator()
    # 可変
    if "df_retrieved" not in st.session_state:
        st.session_state.df_retrieved = pd.DataFrame()
    if "matched_chunk_markdowns" not in st.session_state:
        st.session_state.matched_chunk_markdowns = []
    if "reasons" not in st.session_state:
        st.session_state.reasons = []
    if "query" not in st.session_state:
        st.session_state.query = None
    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = []
    if "file_id" not in st.session_state:
        st.session_state.file_id = "no_file_yet"
    if "n_chunk" not in st.session_state:
        st.session_state.n_chunk = 0
    # モデル選択用
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = cfg.gemini_llm_name


def setup_sidebar():
    """サイドバーにモデル選択機能を追加"""
    with st.sidebar:
#        st.header("⚙️ 設定")

        # モデル選択
        st.subheader("LLMモデル選択")
        selected_model = st.selectbox(
            "使用するGeminiモデル",
            cfg.gemini_models,
            index=cfg.gemini_models.index(st.session_state.selected_model) if st.session_state.selected_model in cfg.gemini_models else 0,
            help="生成タスクに使用するGeminiモデルを選択してください"
        )

        # モデルが変更された場合、Generatorを再初期化
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            cfg.gemini_llm_name = selected_model
            st.session_state.generator = Generator()
            st.success(f"モデルを {selected_model} に変更しました")

        st.divider()
        st.caption(f"現在のモデル: **{st.session_state.selected_model}**")


def main():
    st.set_page_config(layout="wide")
    init_session_state()
    setup_sidebar()

    # ページ定義
    page_1_obj = st.Page(page_1, title="page 1", icon="📄")
    query_detail_obj = st.Page(query_detail, title="類似文献検索結果", icon="🔍")
    search_results_obj = st.Page(search_results_list, title="検索結果一覧", icon="📊")
    ai_judge_obj = st.Page(ai_judge_detail, title="AI審査詳細", icon="⚖️")
    prior_art_obj = st.Page(prior_art_detail, title="先行技術詳細", icon="📑")

    pages = [
        page_1_obj,
        query_detail_obj,
        search_results_obj,
        ai_judge_obj,
        prior_art_obj,
    ]

    # ページオブジェクトをSession Stateに保存し、他のファイルから参照可能にする
    st.session_state.page_map = {p.title: p for p in pages}

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
