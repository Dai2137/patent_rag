"""
アプリケーション設定とパス管理モジュール

プロジェクト全体の設定とディレクトリ構造を一元管理します。
"""
import re
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional


# ==============================================================================
# プロジェクトルート（全ファイルで共通使用）
# ==============================================================================
# このファイルは src/infra/config.py なので、3階層上がプロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = PROJECT_ROOT  # 後方互換性のため残す


# ==============================================================================
# サブディレクトリ名の定数管理
# ==============================================================================
class DirNames(StrEnum):
    """
    サブディレクトリ名を一元管理（新規追加は1行で完結）

    Layer 4（Leaf）のディレクトリ名を定義します。
    新しいディレクトリが必要な場合は、ここに1行追加するだけで全体に反映されます。
    """
    UPLOADED = "uploaded"
    KNOWLEDGE = "knowledge"
    QUERY = "query"
    TOPK = "topk"
    ABSTRACT_CLAIMS = "abstract_claims"
    AI_JUDGE = "ai_judge"
    AI_JUDGE_TABLE = "ai_judge_result_table"
    SEARCH = "search"
    LOGS = "logs"
    CACHE = "cache"
    HIMOTUKI_DOC_CONTENTS = "himotuki_doc_contents"
    DOC_FULL_CONTENT = "doc_full_content"
    EVIDENCE_EXTRACTION = "evidence_extraction"


# ==============================================================================
# PathManager: ディレクトリとファイルパスの一元管理
# ==============================================================================
class PathManager:
    """
    特許RAGシステムにおけるディレクトリ構造とファイルパスを一元管理するクラス。

    4階層ディレクトリ構造:
    [Layer 1] ROOT_PATH   (環境依存: PROJECT_ROOT, /mnt/data など)
      └── [Layer 2] GROUP_NAME  (固定サブ: eval, data_store など)
           └── [Layer 3] <DOC_ID>  (動的: doc_number)
                └── [Layer 4] <DirNames>  (頻繁に変更・追加: uploaded, topk など)

    具体的なディレクトリ構成:
    PROJECT_ROOT/
      ├── eval/                          # 本番データ保存場所
      │    ├── temp/                     # 一時ファイル保存場所（Phase 1）
      │    │    └── temp_query.xml       # アップロード直後の一時ファイル
      │    ├── {doc_number}/             # 特許IDごとのプロジェクトディレクトリ
      │    │    ├── uploaded/            # DirNames.UPLOADED
      │    │    ├── topk/                # DirNames.TOPK
      │    │    ├── ai_judge/            # DirNames.AI_JUDGE
      │    │    └── logs/                # DirNames.LOGS
      │    └── {another_doc_number}/
      └── data_store/                    # ベクトルストア

    2段階保存戦略:
      Phase 1 (Temporary): eval/temp/ に保存してXMLをparseし、doc_numberを取得
      Phase 2 (Permanent): eval/{doc_number}/ に移動して正式保存
    """

    # ==============================================================================
    # Layer 1 & 2: 環境依存のルート設定（環境ごとに変更可能）
    # ==============================================================================
    # 🔧 設定変更箇所: 環境に応じてここを変更
    ROOT_PATH: Path = PROJECT_ROOT         # Layer 1: ルートディレクトリ
    GROUP_NAME: str = "eval"               # Layer 2: グループ名（本番データ）

    # ベースディレクトリ（既存コードとの互換性のため）
    EVAL_DIR = PROJECT_ROOT / "eval"          # 評価・本番データ
    TEMP_DIR = EVAL_DIR / "temp"              # 一時ファイル（evalの下）
    DATA_STORE_DIR = PROJECT_ROOT / "data_store"  # ベクトルストア（後方互換性）
    KNOWLEDGE_DIR = EVAL_DIR / DirNames.KNOWLEDGE  # ナレッジディレクトリ（知識ベース）

    @classmethod
    def setup(cls) -> None:
        """
        基本ディレクトリの作成（アプリケーション起動時に実行）
        """
        cls.TEMP_DIR.mkdir(exist_ok=True, parents=True)
        cls.EVAL_DIR.mkdir(exist_ok=True, parents=True)
        cls.DATA_STORE_DIR.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------------------------------
    # 汎用メソッド: 任意のディレクトリ・ファイルに対応
    # --------------------------------------------------------------------------

    @classmethod
    def get_dir(cls, doc_number: str, dir_name: DirNames | str) -> Path:
        """
        任意のサブディレクトリを取得・作成する汎用メソッド

        4階層構造に従ってパスを構築します:
        ROOT_PATH / GROUP_NAME / doc_number / dir_name

        使い方の例:
            PathManager.get_dir("JP2023-12345", DirNames.TOPK)
            PathManager.get_dir("2023000001", DirNames.UPLOADED)

        Args:
            doc_number: 特許公開番号（例: "2023000001"）
            dir_name: DirNames enum または文字列

        Returns:
            指定されたサブディレクトリの絶対パス
        """
        target_dir = cls.ROOT_PATH / cls.GROUP_NAME / doc_number / str(dir_name)
        target_dir.mkdir(exist_ok=True, parents=True)
        return target_dir

    @classmethod
    def get_file(cls, doc_number: str, dir_name: DirNames | str, filename: str) -> Path:
        """
        任意のファイルパスを取得する汎用メソッド

        使い方の例:
            PathManager.get_file("JP2023-12345", DirNames.TOPK, "result.csv")
            PathManager.get_file("2023000001", DirNames.LOGS, "app.log")

        Args:
            doc_number: 特許公開番号
            dir_name: DirNames enum または文字列
            filename: ファイル名

        Returns:
            指定されたファイルの絶対パス
        """
        return cls.get_dir(doc_number, dir_name) / filename

    # --------------------------------------------------------------------------
    # Phase 1: 一時ファイル操作
    # --------------------------------------------------------------------------

    @classmethod
    def get_temp_path(cls, filename: str) -> Path:
        """
        一時ファイルパスを取得

        Args:
            filename: ファイル名（例: "temp_query.xml"）

        Returns:
            一時ファイルの絶対パス
        """
        return cls.TEMP_DIR / filename

    # --------------------------------------------------------------------------
    # Phase 2: 本番ディレクトリ操作
    # --------------------------------------------------------------------------

    @classmethod
    def get_project_dir(cls, doc_number: str) -> Path:
        """
        特許IDごとのプロジェクトディレクトリを取得（なければ作成）

        Args:
            doc_number: 特許公開番号（例: "2023000001"）

        Returns:
            プロジェクトディレクトリの絶対パス
        """
        if not doc_number:
            raise ValueError("doc_number は空文字列にできません")

        project_dir = cls.EVAL_DIR / doc_number
        project_dir.mkdir(exist_ok=True, parents=True)
        return project_dir

    @classmethod
    def move_to_permanent(
        cls,
        temp_path: Path,
        doc_number: str,
        permanent_filename: str = "uploaded_query.txt"
    ) -> Path:
        """
        一時ファイルを本番ディレクトリへコピー（Phase 2: Permanent）

        ファイルは eval/{doc_number}/uploaded/ 配下に保存されます。

        Args:
            temp_path: 一時ファイルのパス
            doc_number: 特許公開番号
            permanent_filename: 本番ファイル名（デフォルト: "uploaded_query.txt"）

        Returns:
            本番ディレクトリに配置されたファイルの絶対パス
            (例: eval/{doc_number}/uploaded/uploaded_query.txt)

        Raises:
            FileNotFoundError: 一時ファイルが存在しない場合
            ValueError: doc_numberが空の場合
        """
        if not temp_path.exists():
            raise FileNotFoundError(f"一時ファイルが見つかりません: {temp_path}")

        # uploaded ディレクトリを取得・作成
        uploaded_dir = cls.get_dir(doc_number, DirNames.UPLOADED)
        permanent_path = uploaded_dir / permanent_filename

        # ファイルコピー（メタデータ保持、上書き許可）
        shutil.copy2(temp_path, permanent_path)

        return permanent_path

    # --------------------------------------------------------------------------
    # 個別ディレクトリ取得メソッド（便利メソッド）
    # --------------------------------------------------------------------------

    @classmethod
    def get_uploaded_query_path(cls, doc_number: str) -> Path:
        """
        アップロードディレクトリのパスを取得

        Returns:
            eval/{doc_number}/uploaded/
        """
        return cls.get_dir(doc_number, DirNames.UPLOADED)

    @classmethod
    def get_topk_results_path(cls, doc_number: str) -> Path:
        """
        TopK検索結果ディレクトリのパスを取得

        Returns:
            eval/{doc_number}/topk/
        """
        return cls.get_dir(doc_number, DirNames.TOPK)

    @classmethod
    def get_ai_judge_result_path(cls, doc_number: str) -> Path:
        """
        AI審査結果ディレクトリのパスを取得

        Returns:
            eval/{doc_number}/ai_judge/
        """
        return cls.get_dir(doc_number, DirNames.AI_JUDGE)

    @classmethod
    def get_himotuki_doc_contents(cls, doc_number: str) -> Path:
        """
        AI審査結果ディレクトリのパスを取得

        Returns:
            eval/{doc_number}/ai_judge/
        """
        return cls.get_dir(doc_number, DirNames.HIMOTUKI_DOC_CONTENTS)

    # --------------------------------------------------------------------------
    # レガシー対応: インスタンスメソッド版（既存コードとの互換性）
    # --------------------------------------------------------------------------

    def __init__(self, public_doc_number: str):
        """
        インスタンス版コンストラクタ（既存コードとの互換性のため残す）

        Args:
            public_doc_number: 特許の公開番号（例: "2013086509"）
        """
        self.public_doc_number = public_doc_number
        self.base_dir = self.get_project_dir(public_doc_number)

    def get_base_dir(self) -> Path:
        """public_doc_number 専用のベースディレクトリを取得"""
        return self.base_dir

    def ensure_base_dir(self) -> None:
        """ベースディレクトリが存在しない場合は作成（クラスメソッド版で自動実行）"""
        self.base_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Config: アプリケーション全体の設定
# ==============================================================================
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
    persist_dir = str(PathManager.DATA_STORE_DIR / "chroma" / "gemini_v0.2")
    # persist_dir = str(PathManager.DATA_STORE_DIR / "chroma" / "openai_v1.0")

    # LLM
    llm_type = "gemini"  # "openai" or "gemini"
    openai_llm_name = "gpt-5-nano"  # gpt-5-nano（最安）, gpt-5（最高品質）
    gemini_llm_name = "gemini-2.5-flash-lite"

    # Available Gemini models for selection
    gemini_models = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]


# ==============================================================================
# グローバルインスタンス
# ==============================================================================
cfg = Config()

# アプリケーション起動時にディレクトリを作成
PathManager.setup()