"""
Generator（判断根拠生成ロジック）のデバッグ用スクリプト

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

from src.app.generator import Generator
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
    print("Generator デバッグ開始")
    print("=" * 60)

    # 1. Generatorの初期化
    print("\n[Step 1] Generatorを初期化中...")
    generator = Generator()
    print(f"✓ Generatorの初期化完了")
    print(f"  - モデル: {generator.model}")

    # 2. テスト用クエリの読み込み
    print(f"\n[Step 2] テストクエリを読み込み中: {TEST_QUERY_FILE}")
    loader = CommonLoader()
    query_patent: Patent = loader.run(Path(TEST_QUERY_FILE))
    print(f"✓ クエリ読み込み完了")
    print(f"  - 出願番号: {query_patent.publication.doc_number}")
    print(f"  - 発明の名称: {query_patent.invention_title}")

    # 3. 検索実行（判断根拠生成のための関連特許を取得）
    print(f"\n[Step 3] 関連特許を検索中...")
    retriever = Retriever(knowledge_dir=KNOWLEDGE_DIR)
    retrieved_docs = retriever.retrieve(query_patent)
    print(f"✓ 検索完了: {len(retrieved_docs)}件")

    if not retrieved_docs:
        print("⚠ 検索結果が0件のため、判断根拠を生成できません")
        return

    # 4. 判断根拠の生成 ★ここにブレークポイントを置くと、generate()内部をステップ実行できます
    print(f"\n[Step 4] 判断根拠を生成中（1件目のみ）...")
    first_doc = retrieved_docs[0]
    print(f"  対象特許: {first_doc.metadata['publication_number']}")

    reason = generator.generate(query_patent, first_doc)

    # 5. 結果の表示
    print(f"\n[Step 5] 生成結果")
    print("=" * 60)
    print(reason)
    print("=" * 60)

    print("\n✓ デバッグ完了")


if __name__ == "__main__":
    main()
