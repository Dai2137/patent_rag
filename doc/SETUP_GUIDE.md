# Patent RAG セットアップガイド

このドキュメントは、patent_ragプロジェクトの初期セットアップ手順と実施した修正内容をまとめたものです。

## 目次
1. [環境構築](#環境構築)
2. [GitHubリポジトリ設定](#githubリポジトリ設定)
3. [依存パッケージのインストール](#依存パッケージのインストール)
4. [API設定](#api設定)
5. [実施した修正内容](#実施した修正内容)
6. [Web Interfaceの起動](#web-interfaceの起動)

---

## 環境構築

### 1. Python仮想環境（venv）の構築

プロジェクトルートで以下を実行：

```bash
cd /home/sonozuka/staging
python3 -m venv venv
```

### 2. VSCode設定（ターミナル自動venv起動）

VSCodeでターミナルを開くたびに自動的にvenv環境をアクティベートするように設定。

**作成ファイル: `.vscode/settings.json`**

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

**確認方法:**
1. VSCodeで新しいターミナルを開く（Ctrl + `）
2. プロンプトの先頭に `(venv)` が表示されることを確認

---

## GitHubリポジトリ設定

### 1. 元のリポジトリからクローン

```bash
cd /home/sonozuka/staging
git clone git@github.com:kawausodanu/patent_rag.git patent_rag
```

### 2. 新しいリポジトリとして初期化

```bash
cd patent_rag
# 元のGit履歴を削除
rm -rf .git

# 新しいリポジトリとして初期化
git init
git branch -m main

# 初回コミット
git add .
git commit -m "Initial commit: Import patent_rag codebase"
```

### 3. GitHub CLIのインストール

```bash
# リポジトリ追加
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# インストール
sudo apt update
sudo apt install gh

# 認証
gh auth login
# → GitHub.com を選択
# → SSH を選択
# → Login with a web browser を選択
# → 表示されたコードをブラウザで入力
```

### 4. GitHubに新しいリポジトリを作成してプッシュ

```bash
# リポジトリ作成
gh repo create OnozukaRikkyo/patent_rag --public --source=. --remote=origin

# SSH URLに変更
git remote set-url origin git@github.com:OnozukaRikkyo/patent_rag.git

# プッシュ
git push -u origin main
```

**リポジトリURL:** https://github.com/OnozukaRikkyo/patent_rag

---

## 依存パッケージのインストール

### 1. uvパッケージマネージャーの確認

```bash
uv --version
# uv 0.8.22
```

### 2. 依存パッケージのインストール

```bash
cd /home/sonozuka/staging/patent_rag
uv sync
```

**インストールされる主要パッケージ:**
- Streamlit (Web UI)
- LangChain (RAGフレームワーク)
- OpenAI / Google Gemini SDK
- ChromaDB (ベクトルDB)
- pandas, lxml, tqdm など

---

## API設定

### 1. 環境変数ファイルの作成

**作成ファイル: `.env.example`**

```env
# OpenAI API Key
# https://platform.openai.com/api-keys から取得してください
OPENAI_API_KEY=your-openai-api-key-here

# Google Gemini API Key (オプション)
# https://makersuite.google.com/app/apikey から取得してください
GOOGLE_API_KEY=your-google-api-key-here
```

### 2. 実際の.envファイルを作成

```bash
cp .env.example .env
# .envファイルを編集して実際のAPIキーを設定
```

### 3. .gitignoreに.envを追加

```bash
echo ".env" >> .gitignore
```

---

## 実施した修正内容

### 1. src/gui.py の修正

#### 修正内容：
- `.env`ファイルの自動読み込み機能を追加
- サイドバーにモデル選択機能を追加
- 実行中にGeminiモデルを切り替え可能に

**変更箇所:**

```python
# .envファイルの読み込み追加
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# サイドバーにモデル選択機能を追加
def setup_sidebar():
    """サイドバーにモデル選択機能を追加"""
    with st.sidebar:
        st.header("⚙️ 設定")

        st.subheader("LLMモデル選択")
        selected_model = st.selectbox(
            "使用するGeminiモデル",
            cfg.gemini_models,
            index=cfg.gemini_models.index(st.session_state.selected_model) if st.session_state.selected_model in cfg.gemini_models else 0,
            help="生成タスクに使用するGeminiモデルを選択してください"
        )

        # モデルが変更された場合、Generatorを再初期化
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            cfg.gemini_llm_name = selected_model
            st.session_state.generator = Generator()
            st.success(f"モデルを {selected_model} に変更しました")

        st.divider()
        st.caption(f"現在のモデル: **{st.session_state.selected_model}**")
```

### 2. src/infra/config.py の修正

#### 修正内容：
- OpenAIからGemini APIに切り替え
- 利用可能なGeminiモデルのリストを追加
- デフォルトモデルを `gemini-2.5-flash-lite` に設定

**変更箇所:**

```python
@dataclass
class Config:
    # Embeddings, Retriever
    embedding_type = "gemini"  # "openai" → "gemini" に変更
    gemini_embedding_model_name = "models/text-embedding-004"  # 更新

    # Chroma
    persist_dir = "data_store/chroma/gemini_v0.2"  # パスを変更

    # LLM
    llm_type = "gemini"  # "openai" → "gemini" に変更
    gemini_llm_name = "gemini-2.5-flash-lite"  # デフォルトモデル

    # 利用可能なGeminiモデルのリスト（新規追加）
    gemini_models = [
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
```

### 3. .gitignore の更新

```bash
# 追加項目
.env
```

### 4. Streamlit設定ファイルの作成

**作成ファイル: `~/.streamlit/config.toml`**

```toml
[browser]
gatherUsageStats = false

[server]
headless = true
```

---

## Web Interfaceの起動

### 起動方法

```bash
cd /home/sonozuka/staging/patent_rag
uv run streamlit run src/gui.py
```

### アクセスURL

- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.0.3:8501
- **External URL**: http://118.158.74.172:8501

### 停止方法

```bash
pkill -f "streamlit run src/gui.py"
```

---

## 使用するAPI・モデル

### 現在の設定

- **Embedding**: Google Gemini `models/text-embedding-004`
- **LLM**: Google Gemini `gemini-2.5-flash-lite`（デフォルト）
- **Vector DB**: ChromaDB

### モデルの変更方法

1. Web interfaceのサイドバー「⚙️ 設定」セクションを開く
2. 「使用するGeminiモデル」ドロップダウンから選択
3. モデルが即座に切り替わり、成功メッセージが表示される

**選択可能なモデル:**
- gemini-2.5-flash-lite（デフォルト・推奨）
- gemini-2.0-flash-exp
- gemini-1.5-flash
- gemini-1.5-pro

---

## トラブルシューティング

### APIクォータエラーが発生した場合

**エラー:** `Error code: 429 - insufficient_quota`

**対処法:**
1. Google AI Studio (https://aistudio.google.com) の無料モデルを使用
2. `.env`ファイルに `GOOGLE_API_KEY` を設定
3. `config.py` で `embedding_type` と `llm_type` を `"gemini"` に変更

### ベクトルDBの再構築が必要な場合

```bash
# data_store ディレクトリを削除
rm -rf /home/sonozuka/staging/patent_rag/data_store

# アプリを再起動すると自動的に再構築される
uv run streamlit run src/gui.py
```

---

## ディレクトリ構成

```
/home/sonozuka/staging/
├── .vscode/               # VSCode設定
│   └── settings.json      # venv自動起動設定
├── venv/                  # Python仮想環境（staging直下）
└── patent_rag/            # プロジェクト本体（Gitリポジトリ）
    ├── .venv/             # uv管理の仮想環境
    ├── .env               # API キー（.gitignoreで除外）
    ├── .env.example       # API キーのテンプレート
    ├── src/
    │   ├── gui.py         # [修正] .env読み込み、モデル選択UI追加
    │   ├── infra/
    │   │   └── config.py  # [修正] Gemini設定、モデルリスト追加
    │   └── ...
    ├── data_store/        # ベクトルDB永続化領域
    │   └── chroma/
    │       └── gemini_v0.2/
    └── ...
```

---

## まとめ

1. ✅ Python venv環境を構築し、VSCodeで自動起動するよう設定
2. ✅ 元のリポジトリから独立した新しいGitHubリポジトリを作成
3. ✅ 依存パッケージをuvでインストール
4. ✅ Google Gemini APIを使用するよう設定
5. ✅ Web interfaceにモデル選択機能を追加
6. ✅ Streamlit Web interfaceを起動し、動作確認可能な状態に

これで、patent_ragプロジェクトの開発・拡張を開始できます！
