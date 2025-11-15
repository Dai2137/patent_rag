import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# .envファイルを読み込み
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from app.generator import Generator
from app.retriever import Retriever
from infra.config import cfg
from infra.loader.common_loader import CommonLoader
from ui.gui.page1 import page_1
from ui.gui.page2 import page_2
from ui.gui.page3 import page_3
from ui.gui.page99 import page_99

# 定数
# TODO: GUI関連の定数の適切な定義場所を考える。移動する。
# プロジェクトルート (/home/sonozuka/staging/patent_rag) から相対パスを構築
_PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = str(_PROJECT_ROOT / "eval" / "knowledge")


# セッションステート
# TODO: データはRepositoryクラス、処理はRAGクラスなどにラップしたい。
def init_session_state():
    # 不変
    if "loader" not in st.session_state:
        st.session_state.loader = CommonLoader()
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(knowledge_dir=KNOWLEDGE_DIR)
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
        st.header("⚙️ 設定")

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
    pg = st.navigation([page_1, page_2, page_3, page_99])
    pg.run()


if __name__ == "__main__":
    main()
