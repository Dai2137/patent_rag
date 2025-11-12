# 検索処理フロードキュメント

## 概要

このドキュメントは、GUIアプリケーションで「検索」ボタンをクリックした際の処理の流れ、ベクトル検索の仕組み、データの変換プロセスについて詳しく説明します。

## 目次

1. [検索ボタンクリック後の処理フロー](#検索ボタンクリック後の処理フロー)
2. [knowledge_id の作成方法](#knowledge_id-の作成方法)
3. [knowledge_docs と embedding の関連](#knowledge_docs-と-embedding-の関連)
4. [検索対象文献と top-k の設定](#検索対象文献と-top-k-の設定)
5. [ベクトル検索のエントリーポイント](#ベクトル検索のエントリーポイント)

---

## 検索ボタンクリック後の処理フロー

### 全体フロー図

```
【検索ボタンクリック】
    ↓
【1. クエリのロード】
page1.py:74
  st.session_state.loader.run(QUERY_PATH)
  → XMLファイルを Patent オブジェクトに変換
    ↓
【2. ベクトル検索の実行】
page1.py:76
  retrieve(st.session_state.retriever, query)
    ↓
utils.py:23
  retriever.retrieve(query)
    ↓
retriever.py:89-91
  query_str = f"{patent.invention_title}\n{patent.claims[0]}"
  → 検索クエリ文字列の生成
    ↓
retriever.py:107
  self.retriever.invoke(query_str)
  → LangChain による類似度検索（top-k=3）
    ↓
【3. 結果の整形】
utils.py:26-40
  DataFrame 作成（query_id, knowledge_id, retrieved_chunk）
    ↓
【4. 結果の表示】
page1.py:79
  st.dataframe() で表形式表示
```

### 詳細な処理ステップ

#### ステップ1: クエリのロード

**ファイル**: `src/ui/gui/page1.py:74`

```python
query: Patent = st.session_state.loader.run(QUERY_PATH)
```

- `CommonLoader.run()` が呼ばれ、アップロードされた XML ファイル（`data/gui/uploaded_query.txt`）を解析
- `common_loader.py:26-42` で、XML のルート要素のタグに応じて適切なローダー（ST36/ST96 形式）を選択
- `Patent` オブジェクトが生成され、セッション状態に保存

#### ステップ2: 検索の実行

**ファイル**: `src/ui/gui/page1.py:76`

```python
st.session_state.df_retrieved = retrieve(st.session_state.retriever, query)
```

##### 2-1. `retrieve()` 関数（`utils.py:14-40`）

- `retriever.retrieve(query)` を呼び出してベクトル検索を実行

##### 2-2. `Retriever.retrieve()` メソッド（`retriever.py:93-114`）

**検索クエリの生成** (`retriever.py:89-91`)：
```python
query = f"{patent.invention_title}"
query += f"\n{patent.claims[0]}"
```
- 発明のタイトルと第1請求項を連結して検索文字列を生成

**ベクトル検索の実行** (`retriever.py:107`)：
```python
retrieved_docs: list[Document] = self.retriever.invoke(query_str)
```
- LangChain の `Chroma` ベクトルストアで類似度検索
- 設定された `top_n` 件（`retriever.py:24`）の公知例を取得

##### 2-3. 結果の整形（`utils.py:26-40`）

検索結果の `Document` リストから以下の情報を抽出：
- `query_id`: クエリ特許の公開番号
- `knowledge_id`: 公知例の公開番号
- `retrieved_path`: 公知例のファイルパス
- `retrieved_chunk`: マッチしたテキストチャンク

これらを `DataFrame` に変換して返す。

#### ステップ3: 結果の表示

**ファイル**: `src/ui/gui/page1.py:78-79`

```python
if not st.session_state.df_retrieved.empty:
    st.dataframe(st.session_state.df_retrieved[["query_id", "knowledge_id", "retrieved_chunk"]])
```

検索結果が空でなければ、表形式で表示。

---

## knowledge_id の作成方法

`knowledge_id` は、検索結果として表示される公知例特許の公開番号です。以下のフローで作成されます。

### データフロー

#### 1. ベクトルストア構築時（事前準備）

##### 1-1. 公知例の読み込み（`retriever.py:76-82`）

```python
def _load_knowledge(self) -> dict[str, Patent]:
    knowledge = {}
    for path in self.knowledge_paths:
        patent: Patent = self.loader.run(path)
        id: str = patent.publication.doc_number
        knowledge[id] = patent
    return knowledge
```

- `eval/knowledge` ディレクトリ内の各 XML ファイルを `Patent` オブジェクトに変換
- `Patent` オブジェクトは `publication.doc_number` フィールドに**公開番号**を持つ（例: "JP2022043358A"）

##### 1-2. Patent から Document への変換（`patent.py:148-183`）

```python
def to_doc(self) -> Document:
    page_content = ""
    page_content += f"{'\n'.join(self.claims)}\n"  # 請求項
    page_content += f"{self.abstract}\n"           # 要約

    doc = Document(
        page_content=page_content,
        metadata={
            "path": self.path,
            "publication_number": self.publication.doc_number,  # ← ここで設定
            "application_number": self.application.doc_number,
            "invention_title": self.invention_title,
            # ... その他のメタデータ
        },
    )
    return doc
```

**重要**: `metadata["publication_number"]` に公開番号を設定。この `metadata` はベクトルストアに保存され、検索時に一緒に返される。

##### 1-3. ベクトルストアの構築（`retriever.py:66-72`）

```python
knowledge_docs: list[Document] = [patent.to_doc() for patent in self.knowledge.values()]
splitted_docs: list[Document] = splitter.split_documents(knowledge_docs)
chroma = Chroma.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    persist_directory=cfg.persist_dir,
)
```

チャンク分割後も、各チャンクは元の文書の `metadata` を引き継ぐ。

#### 2. 検索実行時（検索ボタンクリック後）

##### 2-1. ベクトル検索（`retriever.py:107`）

```python
retrieved_docs: list[Document] = self.retriever.invoke(query_str)
```

Chroma ベクトルストアから、類似度の高い `Document` オブジェクトのリストを取得。
各 `Document` には `metadata["publication_number"]` が含まれる。

##### 2-2. knowledge_id の抽出（`utils.py:26-30`）

```python
for doc in retrieved_docs:
    query_ids.append(query.publication.doc_number)
    knowledge_ids.append(doc.metadata["publication_number"])  # ← ここで取得
    retrieved_paths.append(doc.metadata["path"])
    retrieved_chunks.append(doc.page_content)
```

検索結果の各 `Document` から `metadata["publication_number"]` を取り出して `knowledge_ids` リストに追加。

##### 2-3. DataFrame の作成（`utils.py:32-40`）

```python
df = pd.DataFrame(
    {
        "query_id": query_ids,
        "knowledge_id": knowledge_ids,  # ← ここで列として設定
        "retrieved_path": retrieved_paths,
        "retrieved_chunk": retrieved_chunks,
    }
)
return df
```

### まとめ

**knowledge_id = 公知例特許の公開番号（publication_number）**

データの流れ：
1. XML から抽出 → `Patent.publication.doc_number` に格納
2. メタデータに保存 → `metadata["publication_number"]` として Document に設定
3. ベクトルストアに保存 → チャンク分割後もメタデータは保持
4. 検索時に取得 → `Document.metadata["publication_number"]` を取り出す
5. DataFrame 化 → `knowledge_id` 列として格納し、画面に表示

---

## knowledge_docs と embedding の関連

### データ変換とベクトル化の全体フロー

#### ステップ1: Patent → Document の変換（`retriever.py:66`）

```python
knowledge_docs: list[Document] = [patent.to_doc() for patent in self.knowledge.values()]
```

**Patent オブジェクト** から **Document オブジェクト** への変換：

**Document オブジェクトの構造**（`patent.py:148-183`）：
```python
def to_doc(self) -> Document:
    page_content = ""
    page_content += f"{'\n'.join(self.claims)}\n"  # 請求項
    page_content += f"{self.abstract}\n"           # 要約

    doc = Document(
        page_content=page_content,  # ← ベクトル化される本文
        metadata={
            "publication_number": self.publication.doc_number,
            "path": self.path,
            # ... その他のメタデータ（ベクトル化されない）
        },
    )
    return doc
```

**重要なポイント**：
- `page_content`: **請求項 + 要約**のテキスト（これが embedding でベクトル化される対象）
- `metadata`: メタデータ（ベクトル化されず、そのまま保存される）

#### ステップ2: チャンク分割（`retriever.py:61-67`）

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.chunk_size,        # 400文字
    chunk_overlap=cfg.chunk_overlap,  # 100文字のオーバーラップ
    add_start_index=True,
)
knowledge_docs: list[Document] = [patent.to_doc() for patent in self.knowledge.values()]
splitted_docs: list[Document] = splitter.split_documents(knowledge_docs)
```

**チャンク分割の例**：
```
元の Document:
page_content = "【請求項1】AはBである装置。【請求項2】CはDである方法。【要約】本発明は..."
                ↓ 400文字ごとに分割（100文字オーバーラップ）
分割後:
- Document1: page_content = "【請求項1】AはBである装置。【請求項2】..." (最初の400文字)
- Document2: page_content = "...【請求項2】CはDである方法。【要約】..." (300文字スキップして次の400文字)
- Document3: page_content = "...本発明は..." (以下同様)
```

**各チャンクは元の metadata を保持**（`publication_number`, `path` など）。

#### ステップ3: Embedding によるベクトル化（`retriever.py:68-72`）

```python
chroma = Chroma.from_documents(
    documents=splitted_docs,
    embedding=embeddings,  # ← ここで使用
    persist_directory=cfg.persist_dir,
)
```

##### Embedding モデルの初期化（`retriever.py:26-44`）

```python
def _init_embeddings(self) -> Embeddings:
    type = cfg.embedding_type.lower()
    if type == "gemini":
        embeddings: Embeddings = GoogleGenerativeAIEmbeddings(
            model=cfg.gemini_embedding_model_name,  # "models/text-embedding-004"
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif type == "openai":
        embeddings: Embeddings = OpenAIEmbeddings(
            model=cfg.openai_embedding_model_name,  # "text-embedding-3-small"
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return embeddings
```

##### ベクトル化の処理（Chroma 内部で実行）

`Chroma.from_documents()` は内部で以下の処理を実行：

1. **各チャンクのテキストをベクトル化**：
   ```python
   for doc in splitted_docs:
       text = doc.page_content  # "【請求項1】AはBである装置。..."
       vector = embeddings.embed_query(text)  # テキスト → ベクトル（例: 768次元の数値配列）
       # ベクトルとメタデータを Chroma に保存
   ```

2. **保存される情報**：
   - **ベクトル**: `[0.123, -0.456, 0.789, ...]` (768次元など)
   - **元のテキスト**: `doc.page_content`
   - **メタデータ**: `doc.metadata` (publication_number, path など)

#### ステップ4: as_retriever でレトリーバー化（`retriever.py:24`）

```python
self.retriever = self.chroma.as_retriever(search_kwargs={"k": cfg.top_n})
```

`as_retriever()` は、Chroma ベクトルストアを**検索可能なレトリーバーオブジェクト**に変換：
- `search_kwargs={"k": 3}`: 上位3件の類似文書を返す設定
- 検索時には、クエリテキストも同じ embedding モデルでベクトル化され、コサイン類似度などで比較される

### 検索時の処理フロー（`retriever.py:107`）

```python
retrieved_docs: list[Document] = self.retriever.invoke(query_str)
```

内部では以下が実行される：

1. **クエリのベクトル化**：
   ```python
   query_vector = embeddings.embed_query(query_str)  # "発明の名称\n請求項1..." → ベクトル
   ```

2. **類似度検索**：
   - Chroma が保存されている全ベクトルと、クエリベクトルの類似度を計算
   - 類似度が高い上位3件（`k=3`）のチャンクを取得

3. **Document の復元**：
   - ベクトルストアから対応する `page_content` と `metadata` を取得
   - `Document` オブジェクトとして返す

### データフローの全体像

```
Patent オブジェクト
  ↓ .to_doc()
Document (page_content="請求項+要約", metadata={...})
  ↓ split_documents()
複数のチャンク [Document1, Document2, Document3, ...]
  ↓ Chroma.from_documents(embedding=embeddings)
  ├─ page_content → embeddings.embed_query() → ベクトル化 → Chromaに保存
  └─ metadata → そのまま保存

【検索時】
クエリテキスト
  ↓ embeddings.embed_query()
クエリベクトル
  ↓ 類似度計算
Chromaから上位k件取得
  ↓
retrieved_docs: list[Document]
```

### 重要なポイント

1. **`page_content` だけがベクトル化される**: メタデータは検索対象ではなく、結果と一緒に返される付加情報

2. **同じ embedding モデルを使用**: 保存時と検索時で同じモデル（Gemini/OpenAI）を使わないと、ベクトル空間が異なり正しく検索できない

3. **チャンク分割の意義**：
   - 長い文書を分割することで、より細かい粒度でマッチング可能
   - 各チャンクが独立して検索可能になる

4. **persist_directory の重要性**: 一度ベクトル化すれば `config.py:19` のディレクトリに保存され、次回起動時は `retriever.py:54` で読み込むだけで済む（再計算不要）

---

## 検索対象文献と top-k の設定

### 検索対象文献の取得

#### 1. 文献パスの取得（`retriever.py:19`）

```python
def __init__(self, knowledge_dir: str):
    self.knowledge_paths = list(Path(knowledge_dir).rglob("text.txt"))
```

**`rglob("text.txt")`** は再帰的にディレクトリを探索し、全ての `text.txt` ファイルのパスを取得。

#### 2. knowledge_dir の指定（`gui.py:22-32`）

```python
# 定数
KNOWLEDGE_DIR = "eval/knowledge"

def init_session_state():
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(knowledge_dir=KNOWLEDGE_DIR)
```

**`"eval/knowledge"`** ディレクトリが検索対象として指定される。

#### 3. ディレクトリ構造

```
eval/knowledge/
├── result_1/
│   └── 0/
│       ├── JP2010000001A/
│       │   ├── text.txt     ← 検索対象
│       │   ├── text.xml
│       │   ├── text.json
│       │   └── JP2010000001A.pdf
│       └── JP2015039043A/
│           └── text.txt      ← 検索対象
├── result_2/
│   └── ...
└── result_3/
    └── ...
```

現在のプロジェクトでは**合計42件**の `text.txt` ファイルが検索対象。

#### 4. 文献の読み込み（`retriever.py:76-82`）

```python
def _load_knowledge(self) -> dict[str, Patent]:
    knowledge = {}
    for path in self.knowledge_paths:
        patent: Patent = self.loader.run(path)
        id: str = patent.publication.doc_number
        knowledge[id] = patent
    return knowledge
```

各 `text.txt` ファイル（XML ファイル）を読み込み、`Patent` オブジェクトに変換。

### top-k の設定

#### 1. 設定ファイルでの定義（`config.py:16`）

```python
@dataclass
class Config:
    # Embeddings, Retriever
    embedding_type = "gemini"
    openai_embedding_model_name = "text-embedding-3-small"
    gemini_embedding_model_name = "models/text-embedding-004"
    chunk_size = 400
    chunk_overlap = 100
    top_n = 3  # ← ここで定義
```

**`top_n = 3`** がデフォルト値として設定される。

#### 2. Retriever での使用（`retriever.py:24`）

```python
self.retriever = self.chroma.as_retriever(search_kwargs={"k": cfg.top_n})
```

**`search_kwargs={"k": cfg.top_n}`** によって、検索時に**上位3件**を返すように設定。

#### 3. 検索実行時（`retriever.py:107`）

```python
def retrieve(self, query: str | Patent) -> list[Document]:
    # ...
    retrieved_docs: list[Document] = self.retriever.invoke(query_str)
    return retrieved_docs
```

`self.retriever.invoke()` を呼び出すと、内部で：
1. クエリをベクトル化
2. Chroma ベクトルストア内の全チャンクと類似度を計算
3. **類似度が高い上位3件**（`k=3`）のチャンクを返す

### 設定の変更方法

#### 検索対象ディレクトリを変更する場合

`src/gui.py:22` を編集：
```python
# 例: 別のディレクトリを指定
KNOWLEDGE_DIR = "data/my_patents"

# 例: 環境変数から取得
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "eval/knowledge")
```

#### top-k を変更する場合

`src/infra/config.py:16` を編集：
```python
top_n = 5  # 上位5件を取得
```

### まとめ表

| 項目 | 定義場所 | 値 |
|------|---------|-----|
| **検索対象文献の指定** | `gui.py:22` | `"eval/knowledge"` |
| **文献パスの取得** | `retriever.py:19` | `Path(knowledge_dir).rglob("text.txt")` |
| **文献数** | 実行時確認 | 42件 |
| **top-k の定義** | `config.py:16` | `top_n = 3` |
| **top-k の使用** | `retriever.py:24` | `search_kwargs={"k": cfg.top_n}` |
| **検索実行** | `retriever.py:107` | `self.retriever.invoke(query_str)` |

---

## ベクトル検索のエントリーポイント

ベクトル検索は複数の階層で呼び出されます。

### 第1階層: UI レベルのエントリー（`page1.py:73-76`）

```python
if st.button("検索", type="primary"):
    query: Patent = st.session_state.loader.run(QUERY_PATH)
    st.session_state.query = query
    st.session_state.df_retrieved = retrieve(st.session_state.retriever, query)
```

- **イベント**: `st.button()` がクリックされたときに `True` を返す
- **関数**: `retrieve()` （utils.py で定義）
- **役割**: ボタンクリックイベントから検索を開始
- **引数**：
  - `st.session_state.retriever`: 初期化済みの Retriever インスタンス
  - `query`: アップロードされた XML から生成した Patent オブジェクト

### 第2階層: アプリケーションレベルのエントリー（`utils.py:14-24`）

```python
def retrieve(retriever: Retriever, query: Patent) -> pd.DataFrame:
    """
    検索を実行して、検索結果を返す
    """
    query_ids: list[str] = []
    knowledge_ids: list[str] = []
    retrieved_paths: list[str] = []
    retrieved_chunks: list[str] = []

    retrieved_docs: list[Document] = retriever.retrieve(query)  # ← エントリー
    st.session_state.retrieved_docs = retrieved_docs
    # ... 以下、DataFrame作成処理
```

- **関数**: `Retriever.retrieve()` （retriever.py のメソッド）
- **役割**: UI 層とドメイン層の橋渡し、結果の DataFrame 化

### 第3階層: ドメインレベルのエントリー（`retriever.py:93-107`）

```python
def retrieve(self, query: str | Patent) -> list[Document]:
    """
    新規出願特許（query）に関連する公開特許を返す。
    """
    if isinstance(query, str):
        query_str = query
    elif isinstance(query, Patent):
        query_str = self._to_str(query)  # タイトル + 請求項1 に変換
    else:
        raise ValueError("クエリは、strかPatent型にしてください。")

    # ベクトル検索
    retrieved_docs: list[Document] = self.retriever.invoke(query_str)  # ← エントリー

    return retrieved_docs
```

- **メソッド**: `self.retriever.invoke()` （LangChain のレトリーバー）
- **役割**: Patent オブジェクトを検索用文字列に変換し、ベクトル検索を実行

### 第4階層: インフラレベル（LangChain 内部）（`retriever.py:107`）

```python
retrieved_docs: list[Document] = self.retriever.invoke(query_str)
```

LangChain のレトリーバー（`self.retriever` は `retriever.py:24` で作成）が内部で実行：

1. `embeddings.embed_query(query_str)` でクエリをベクトル化
2. Chroma ベクトルストアで類似度検索（コサイン類似度など）
3. 上位 k 件（`k=3`）の Document を返す

### データフロー全体図

```
【検索ボタンクリック】
page1.py:73
  st.button("検索")
  ↓ クリックイベント発火
page1.py:76
  retrieve(retriever, query) ← 【第1階層: UIエントリー】
  ↓
utils.py:23
  retriever.retrieve(query) ← 【第2階層: アプリケーションエントリー】
  ↓
retriever.py:93-102
  query_str = self._to_str(query)
  → "発明の名称\n【請求項1】..."
  ↓
retriever.py:107
  self.retriever.invoke(query_str) ← 【第3階層: ドメインエントリー】
  ↓
【LangChain内部】← 【第4階層: インフラエントリー】
  1. embeddings.embed_query(query_str) → クエリベクトル
  2. chroma.similarity_search(vector, k=3) → 類似度検索
  3. return [Document, Document, Document] (top-3)
  ↓
utils.py:26-40
  DataFrame作成（query_id, knowledge_id, retrieved_chunkの列）
  ↓
page1.py:79
  st.dataframe() で表示
```

### 各階層の責務

| 階層 | エントリーポイント | 責務 |
|------|-------------------|------|
| **第1階層（UI）** | `page1.py:76`<br>`retrieve(retriever, query)` | ユーザーインタラクションの処理 |
| **第2階層（アプリケーション）** | `utils.py:23`<br>`retriever.retrieve(query)` | UI層とドメイン層の橋渡し、データ変換 |
| **第3階層（ドメイン）** | `retriever.py:107`<br>`self.retriever.invoke(query_str)` | ビジネスロジック、検索クエリ生成 |
| **第4階層（インフラ）** | LangChain 内部 | ベクトル化、類似度計算、ストレージアクセス |

---

## 参考情報

### 関連ファイル一覧

- **UI層**:
  - `src/ui/gui/page1.py`: 検索ボタンとイベントハンドラ
  - `src/ui/gui/utils.py`: 検索結果の DataFrame 化
  - `src/gui.py`: アプリケーション初期化、セッション状態管理

- **ドメイン層**:
  - `src/app/retriever.py`: ベクトル検索のメインロジック
  - `src/model/patent.py`: Patent データモデル、Document 変換

- **インフラ層**:
  - `src/infra/loader/common_loader.py`: XML ファイルのロード
  - `src/infra/config.py`: アプリケーション設定

### 設定ファイル

`src/infra/config.py`:
```python
@dataclass
class Config:
    # Embeddings, Retriever
    embedding_type = "gemini"
    openai_embedding_model_name = "text-embedding-3-small"
    gemini_embedding_model_name = "models/text-embedding-004"
    chunk_size = 400
    chunk_overlap = 100
    top_n = 3

    # Chroma
    persist_dir = "data_store/chroma/gemini_v0.2"
```

### ベクトルストアの永続化

ベクトルストアは以下のディレクトリに保存されます：
- Gemini: `data_store/chroma/gemini_v0.2/`
- OpenAI: `data_store/chroma/openai_v1.0/`

一度構築されたベクトルストアは永続化され、次回起動時に再利用されます。

---

## 補足: 検索精度の向上方法

1. **top_n の調整**: `config.py` の `top_n` を増やすと、より多くの候補を取得できる

2. **chunk_size の調整**: より細かいチャンク（例: 200文字）にすると、精密なマッチングが可能

3. **検索クエリの改善**: `retriever.py:89-91` でタイトルと請求項1のみ使用しているが、より多くの情報を含めることも可能

4. **embedding モデルの変更**: Gemini と OpenAI で精度が異なる場合があるため、試験的に切り替えてみる

5. **ベクトルストアの再構築**: 新しい文献を追加した場合は、`persist_dir` を削除してベクトルストアを再構築する
