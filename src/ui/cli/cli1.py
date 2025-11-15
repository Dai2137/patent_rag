"""
コマンドライン実行用のメソッドを定義します。
ここで定義されたメソッドはmain.pyから呼び出されます。
"""

from pathlib import Path

import pandas as pd

from app.generator import Generator
from app.rag import Rag
from app.retriever import Retriever


def test_retriever():
    # RAG実行
    retriever = Retriever(knowledge_dir="eval/knowledge/result_1/0")
    generator = Generator()
    rag = Rag(retriever, generator)

    query_paths = list(Path("eval/query/result_4").rglob("text.txt"))
    query_ids, knowledge_ids, reasons = rag.run_retriever(query_paths)

    # CSV出力（ディレクトリが存在しない場合は作成）
    df = pd.DataFrame({"query_id": query_ids, "knowledge_id": knowledge_ids, "reason": reasons})
    output_path = Path("eval/rag_output.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
