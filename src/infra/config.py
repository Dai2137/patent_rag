from dataclasses import dataclass
from pathlib import Path

# プロジェクトルート (/home/sonozuka/staging/patent_rag) を取得
_PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class Config:
    """
    アプリケーション全体の設定を保持するクラス。
    """

    # Embeddings, Retriever
    embedding_type = "gemini"  # "openai" or "gemini"
    openai_embedding_model_name = "text-embedding-3-small"
    gemini_embedding_model_name = "models/text-embedding-004"
    chunk_size = 400
    chunk_overlap = 100
    top_n = 3

    # Chroma - プロジェクトルートからの絶対パスを使用
    persist_dir = str(_PROJECT_ROOT / "data_store" / "chroma" / "gemini_v0.2")
    # persist_dir = str(_PROJECT_ROOT / "data_store" / "chroma" / "openai_v1.0")

    # LLM
    llm_type = "gemini" # "openai" or "gemini"
    openai_llm_name = "gpt-5-nano" # gpt-5-nano（最安）, gpt-5（最高品質）
    gemini_llm_name = "gemini-2.5-flash-lite"

    # Available Gemini models for selection
    gemini_models = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]


cfg = Config()
