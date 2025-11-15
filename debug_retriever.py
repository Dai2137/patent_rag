"""
Retriever（検索ロジック）のデバッグ用スクリプト

使い方：
1. このファイルをVSCodeで開く
2. デバッグしたい行（★マークの行など）にブレークポイントを設定
3. F5キーを押して「Python デバッガー: 現在のファイル」を選択
4. ステップ実行でロジックを確認
"""

from pathlib import Path
from dotenv import load_dotenv

# .envファイルを読み込み
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.app.retriever import Retriever
from src.infra.loader.common_loader import CommonLoader
from src.model.patent import Patent

# 定数
# プロジェクトルート (/home/sonozuka/staging/patent_rag) から相対パスを構築
_PROJECT_ROOT = Path(__file__).parent
KNOWLEDGE_DIR = str(_PROJECT_ROOT / "eval" / "knowledge")
TEST_QUERY_FILE = str(_PROJECT_ROOT / "eval" / "knowledge" / "result_1" / "0" / "JP2010000001A" / "text.txt")


def main():
    print("=" * 60)
    print("Retriever デバッグ開始")
    print("=" * 60)

    # 1. Retrieverの初期化
    print("\n[Step 1] Retrieverを初期化中...")
    retriever = Retriever(knowledge_dir=KNOWLEDGE_DIR)
    print(f"✓ Retrieverの初期化完了")

    # 2. テスト用クエリの読み込み
    print(f"\n[Step 2] テストクエリを読み込み中: {TEST_QUERY_FILE}")
    loader = CommonLoader()
    query_patent: Patent = loader.run(Path(TEST_QUERY_FILE))
    print(f"✓ クエリ読み込み完了")
    print(f"  - 出願番号: {query_patent.publication.doc_number}")
    print(f"  - 発明の名称: {query_patent.invention_title}")

    # 3. 検索実行 ★ここにブレークポイントを置くと、retrieve()内部をステップ実行できます
    print(f"\n[Step 3] 類似特許を検索中...")
    retrieved_docs = retriever.retrieve(query_patent)

    # 4. 結果の表示
    print(f"\n[Step 4] 検索結果: {len(retrieved_docs)}件")
    print("=" * 60)
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n【結果 {i}】")
        print(f"  公開番号: {doc.metadata['publication_number']}")
        print(f"  パス: {doc.metadata['path']}")
        print(f"  内容（先頭100文字）: {doc.page_content[:100]}...")

    print("\n" + "=" * 60)
    print("デバッグ完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
