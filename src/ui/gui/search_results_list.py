import streamlit as st
import pandas as pd
from pathlib import Path
from ui.gui import page1


def search_results_list():
    """検索結果一覧ページ"""

    # session_stateから検索結果を取得
    if 'search_results_df' not in st.session_state or st.session_state.search_results_df is None:
        st.error("❌ 検索結果が見つかりません。")
        if st.button("メインページに戻る"):
            st.switch_page(st.Page(page1.page_1))
        return

    search_results_df = st.session_state.search_results_df
    output_csv_path = st.session_state.get('search_results_csv_path', None)

    # タイトル
    st.title("🔍 検索結果一覧")

    # メインページに戻るボタン
    if st.button("⬅️ メインページに戻る"):
        st.switch_page(st.Page(page1.page_1))

    st.markdown("---")

    # 統計情報の表示
    if len(search_results_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("検索結果件数", f"{len(search_results_df):,}")
        with col2:
            st.metric("最大類似度", f"{search_results_df['cosine_similarity'].max():.4f}")
        with col3:
            st.metric("平均類似度", f"{search_results_df['cosine_similarity'].mean():.4f}")
        with col4:
            st.metric("最小類似度", f"{search_results_df['cosine_similarity'].min():.4f}")

        st.markdown("---")

        # フィルタリングオプション
        st.subheader("🔧 フィルタリング")
        col1, col2 = st.columns(2)

        with col1:
            min_similarity = st.slider(
                "最小類似度",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )

        with col2:
            max_results = st.number_input(
                "表示件数（最大）",
                min_value=10,
                max_value=len(search_results_df),
                value=min(100, len(search_results_df)),
                step=10
            )

        # フィルタリング適用
        filtered_df = search_results_df[search_results_df['cosine_similarity'] >= min_similarity]
        filtered_df = filtered_df.head(max_results)

        st.markdown("---")

        # 結果表示
        st.subheader(f"📋 検索結果（{len(filtered_df):,}件表示）")

        # データフレームをスクロール可能な形式で表示
        st.dataframe(
            filtered_df,
            width="stretch",
            height=600,
            hide_index=False
        )

        st.markdown("---")

        # CSVダウンロードボタン
        if output_csv_path:
            csv_data = search_results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 全結果をCSVダウンロード",
                data=csv_data,
                file_name=Path(output_csv_path).name,
                mime="text/csv"
            )
            st.info(f"💾 結果は {output_csv_path} に保存されています")
    else:
        st.warning("検索結果がありません。")


if __name__ == "__main__":
    search_results_list()