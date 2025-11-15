import streamlit as st
import sys
from pathlib import Path

# big_query_topk.pyをインポートできるようにパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent / "bigquery"))
from big_query_topk import search_similar_patents


def page_3():
    st.write("page3です")

    # 特許番号の入力フィールド
    st.subheader("類似特許検索")
    target_patent = st.text_input(
        "基準とする特許番号を入力してください",
        value="JP-S4926374-B1",
        help="例: JP-2023123456-A, JP-S4926374-B1"
    )

    # Top-K件数の設定
    top_k = st.number_input(
        "取得する類似特許の件数",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100
    )

    # CSVファイル名の設定
    output_csv = f"similar_patents_{target_patent}.csv"

    # クエリーボタンを表示
    if st.button("クエリー"):
        if not target_patent:
            st.error("特許番号を入力してください")
        else:
            try:
                with st.spinner(f"検索中... 特許番号: {target_patent}"):
                    # BigQueryクエリの実行
                    results_df = search_similar_patents(
                        target_patent_number=target_patent,
                        output_csv=output_csv,
                        top_k=top_k
                    )

                    st.success(f"検索完了！{len(results_df)}件の類似特許を発見しました")

                    # 統計情報の表示
                    if len(results_df) > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("最大類似度", f"{results_df['cosine_similarity'].max():.4f}")
                        with col2:
                            st.metric("平均類似度", f"{results_df['cosine_similarity'].mean():.4f}")
                        with col3:
                            st.metric("最小類似度", f"{results_df['cosine_similarity'].min():.4f}")

                        # 結果のテーブル表示
                        st.subheader("Top 10 類似特許")
                        st.dataframe(results_df.head(10))

                        # CSVダウンロードボタン
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="CSVダウンロード",
                            data=csv,
                            file_name=output_csv,
                            mime="text/csv"
                        )

                        st.info(f"結果は {output_csv} に保存されました")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.exception(e)
