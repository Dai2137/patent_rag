import streamlit as st
from pathlib import Path
import pandas as pd
from llm.llm_data_loader import entry

# プロジェクトルート（このファイルは src/ui/gui/ にあるので4階層上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# page1と同じQUERY_PATHを使用
QUERY_PATH = PROJECT_ROOT / "data" / "gui" / "uploaded_query.txt"
    # CSVファイル名の設定
OUTPUT_CSV_PATH = PROJECT_ROOT / "eval" / "ai_judge"

def ai_judge_detail():
    """AI審査結果の詳細画面"""
        # llm_data_loaderのentryを呼び出す
    result = entry()

    # 結果を表示
    st.subheader("llm_data_loader.entry の結果")
    st.json(result['comparison'])
