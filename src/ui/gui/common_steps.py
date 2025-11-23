"""
共通ステップ処理モジュール
新規アップロード・既存プロジェクトの両方で使用される
Step 2（検索）、Step 3（AI審査）、Step 4（根拠生成）の処理を含む
"""

import streamlit as st
import pandas as pd

from infra.config import PathManager
from ui.gui import query_detail, ai_judge_detail
from ui.gui.search_results_list import search_results_list
from ui.gui.prior_art_detail import prior_art_detail


def render_common_steps():
    """
    Step 2以降の共通処理
    新規アップロード後も、既存ロード後も、この関数でUIを描画する
    """

    # --- Step 2: 類似文献検索 ---
    st.header("2. 類似文献の検索")
    st.write("出願の公開番号（query_id）について、Google Patents Public Dataの埋め込みベクトルを用いて類似文献を検索し、上位の文献を表示します。")
    st.write("Google Patents Public Dataは、高精度かつ効率のよい埋め込みベクトルを提供しており、特許文献の意味的な類似性を捉えることができます。")
    st.write("このため、独自に膨大な文献のベクトル化が不要となり、コスト的に効率的な検索が可能です。")

    has_search_results = 'search_results_df' in st.session_state and st.session_state.search_results_df is not None

    if has_search_results:
        search_results_df = st.session_state.search_results_df
        st.info(f"💾 検索結果: {len(search_results_df):,}件 取得済み")

        if st.button("📋 詳細リストを表示", key="goto_search_list"):
            st.switch_page(search_results_list)
        if st.button("🔄 検索をやり直す", key="rerun_search"):
            if "query" not in st.session_state or st.session_state.query is None:
                st.warning("⚠️ 検索を実行するには、先にファイルをアップロードしてください。")
            else:
                query_detail.query_detail()
    else:
        if st.button("検索実行", type="primary", key="run_new_search"):
            if "query" not in st.session_state or st.session_state.query is None:
                st.warning("⚠️ 先にファイルをアップロードしてください。")
            else:
                query_detail.query_detail()

    # --- Step 3: AI審査 ---
    st.header("3. AI審査")
    st.write("大規模言語モデルを活用し、類似度の高い先行技術文献に基づいてAI審査を実行します。")
    st.write("審査では、各先行技術文献が出願に対して新規性・進歩性を欠くかどうかを判断し、判定結果を示します。")
    st.write("課題と解決方法、申請、審査、判定の各専門的な知識を組み合わせ、高精度な審査を目指します。")

    has_ai_results = 'ai_judge_results' in st.session_state and st.session_state.ai_judge_results

    if has_ai_results:
        results = st.session_state.ai_judge_results
        st.info(f"💾 審査結果: {len(results)}件 取得済み")

        # 結果リストの表示
        with st.expander("📋 審査結果一覧", expanded=True):
            for idx, result in enumerate(results):
                if isinstance(result, dict) and 'error' in result:
                    st.error(f"先行技術 #{idx+1}: エラーが発生しました")
                    continue

                doc_num = result.get('prior_art_doc_number', f"先行技術 #{idx+1}")
                st.write(f"**{idx+1}. {doc_num}**")
                if st.button("詳細", key=f"ai_detail_{idx}"):
                    st.session_state.selected_prior_art_idx = idx
                    st.switch_page(prior_art_detail)

        if st.button("🔄 AI審査をやり直す", key="rerun_ai_judge"):
            run_ai_judge()
    else:
        if st.button("AI審査実行", type="primary", key="run_ai_judge_new"):
            # 検索結果がないと実行できない
            if not has_search_results:
                st.warning("⚠️ 先に「2. 類似文献の検索」を実行してください。")
            else:
                run_ai_judge()

    # --- Step 4: 判断根拠出力 ---
    st.header("4. 判断根拠出力")

    # 前提条件チェック
    if not has_ai_results:
        st.write("⚠️ AI審査を実行すると表示されます。")
    else:
        # AI審査結果の件数をデフォルト値として使用
        n_chunk_default = len(st.session_state.ai_judge_results)

        if st.button("根拠テキスト生成", type="primary", key="generate_reasons"):
            if "retrieved_docs" not in st.session_state or not st.session_state.retrieved_docs:
                st.error("❌ 文献データ(retrieved_docs)がメモリにありません。再検索が必要な可能性があります。")
            else:
                generate_reasons(n_chunk_default)

        if "reasons" in st.session_state and st.session_state.reasons:
            for i, reason in enumerate(st.session_state.reasons):
                st.markdown(f"##### 判断根拠 {i + 1}")
                st.code(reason, language="markdown")


def run_ai_judge():
    """AI審査実行ラッパー"""
    if "df_retrieved" in st.session_state and st.session_state.df_retrieved is not None:
        st.session_state.n_topk = len(st.session_state.df_retrieved)

    with st.spinner("審査プロセスを実行中..."):
        results = ai_judge_detail.entry(action="button_click")
        if results:
            st.session_state.ai_judge_results = results
            st.success("✅ AI審査が完了しました。")
            st.rerun()  # 状態反映のためリロード


def generate_reasons(n_chunk):
    """根拠生成ロジック"""
    st.session_state.reasons = []
    status_text = st.empty()
    progress = st.progress(0)

    # インデックスエラーを防ぐため、実際の配列長とn_chunkの小さい方を取る
    actual_limit = min(n_chunk, len(st.session_state.retrieved_docs))

    for i in range(actual_limit):
        status_text.text(f"{i + 1} / {actual_limit} 件目を生成中です...")
        # generatorがsession_stateにある前提
        if "generator" in st.session_state:
            reason = st.session_state.generator.generate(
                st.session_state.query,
                st.session_state.retrieved_docs[i]
            )
            st.session_state.reasons.append(reason)
        else:
            st.error("❌ Generatorが初期化されていません。")
            break
        progress.progress((i + 1) / actual_limit)

    status_text.text("✅ 生成が完了しました。")