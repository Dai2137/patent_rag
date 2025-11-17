import streamlit as st
from pathlib import Path

from bigquery.big_query_topk import search_similar_patents
from ui.gui.utils import format_patent_number_for_bigquery
import inspect
import pandas as pd

# プロジェクトルート（このファイルは src/ui/gui/ にあるので4階層上）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# page1と同じQUERY_PATHを使用
QUERY_PATH = PROJECT_ROOT / "eval" / "uploaded" / "uploaded_query.txt"
    # CSVファイル名の設定
OUTPUT_CSV_PATH = PROJECT_ROOT / "eval" / "topk"
    
def query_detail():
        # 呼び出し元の情報を取得
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    
    # 呼び出し元の関数名
    # step2から呼び出される
    expected_caller_name = "step2"
    search_button_pressed = False
    caller_name = caller_frame.f_code.co_name
    if caller_name == expected_caller_name:
        search_button_pressed = True
    print(caller_name)

    """検索結果の詳細画面"""
    st.write("検索結果詳細")

    # アップロードされた特許番号を取得
    query_patent_number = ""

    # アップロード済みファイルがあれば読み込む
    if QUERY_PATH.exists() and hasattr(st.session_state, 'loader'):
        try:
            query = st.session_state.loader.run(QUERY_PATH)
            query_patent_number = format_patent_number_for_bigquery(query)
        except Exception as e:
            st.warning(f"保存されたファイルからの特許番号取得に失敗しました: {e}")

    # 特許番号の入力フィールド
    st.subheader("類似特許検索の詳細設定")
    query_patent = st.text_input(
        "基準とする特許番号を入力してください",
        value=query_patent_number,
        help="例: JP-2023123456-A, JP-S4926374-B1"
    )

    # Top-K件数の設定
    top_k_count = st.number_input(
        "取得する類似特許の件数",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100
    )
    
    output_csv_path =  OUTPUT_CSV_PATH / f"{query_patent}.csv"

    # 検索実行ボタン
    if  not search_button_pressed:
        if not output_csv_path.exists:
            return
        search_results_df = pd.read_csv(output_csv_path)
        show_result(search_results_df, output_csv_path)
    else:

        if not query_patent:
            st.error("特許番号を入力してください")
        else:

            try:
                with st.spinner(f"検索中... 特許番号: {query_patent}"):
                    # BigQueryクエリの実行
                    search_results_df = search_similar_patents(
                        target_patent_number=query_patent,
                        output_csv=output_csv_path,
                        top_k=top_k_count
                    )

                    st.success(f"検索完了！{len(search_results_df)}件の類似特許を発見しました")
                    show_result(search_results_df, output_csv_path)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.exception(e)
                raise e

def show_result(search_results_df, output_csv_path):
    # 統計情報の表示
    if len(search_results_df) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最大類似度", f"{search_results_df['cosine_similarity'].max():.4f}")
        with col2:
            st.metric("平均類似度", f"{search_results_df['cosine_similarity'].mean():.4f}")
        with col3:
            st.metric("最小類似度", f"{search_results_df['cosine_similarity'].min():.4f}")

        # 詳細結果のテーブル表示
        st.subheader("類似特許の詳細（Top 10）")
        st.dataframe(search_results_df.head(10))

        # 全件表示オプション
        if st.checkbox("全結果を表示"):
            st.subheader(f"全検索結果（{len(search_results_df)}件）")
            st.dataframe(search_results_df)

        # CSVダウンロードボタン
        csv_data = search_results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="CSVダウンロード",
            data=csv_data,
            file_name=Path(output_csv_path).name,
            mime="text/csv"
        )

        st.info(f"結果は {output_csv_path} に保存されました")

