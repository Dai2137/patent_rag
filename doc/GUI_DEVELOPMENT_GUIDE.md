# Streamlit GUI開発ガイド

このドキュメントでは、patent_ragプロジェクトのStreamlit GUIアプリケーションの開発方法について、画面デザイン、ボタンの振る舞い、データフローなど、開発者が実施するすべての作業を具体的に説明します。

## 目次

1. [アーキテクチャ全体像](#アーキテクチャ全体像)
2. [作業1: プロジェクトセットアップ](#作業1-プロジェクトセットアップ)
3. [作業2: 画面デザインの定義](#作業2-画面デザインの定義)
4. [作業3: ボタンの振る舞いの定義](#作業3-ボタンの振る舞いの定義)
5. [作業4: データフローの定義](#作業4-データフローの定義)
6. [作業5: ビジネスロジックとの連携](#作業5-ビジネスロジックとの連携)
7. [作業6: スタイルとカスタマイズ](#作業6-スタイルとカスタマイズ)
8. [作業7: アプリケーションの実行](#作業7-アプリケーションの実行)
9. [開発チェックリスト](#開発チェックリスト)

---

## アーキテクチャ全体像

### Streamlitのアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│              ブラウザー (Client Side)                      │
├─────────────────────────────────────────────────────────┤
│  HTML/CSS (動的生成)                                      │
│    ├─ <button id="button-xxx">検索</button>             │
│    ├─ <div class="stDataFrame">...</div>                │
│    └─ <div class="stMarkdown">...</div>                 │
│                                                          │
│  JavaScript (React + Streamlit Client)                  │
│    ├─ イベントリスナー (ボタンクリック検知)                 │
│    ├─ WebSocket通信管理                                  │
│    └─ DOM更新・レンダリング                               │
└─────────────────────────────────────────────────────────┘
                        ↕ WebSocket (双方向通信)
┌─────────────────────────────────────────────────────────┐
│           Streamlit Server (Server Side)                │
├─────────────────────────────────────────────────────────┤
│  Python Script (gui.py, page1.py, ...)                 │
│    ├─ st.button("検索", type="primary")                 │
│    ├─ st.dataframe(...)                                 │
│    └─ st.session_state.xxx                             │
│                                                          │
│  Streamlit Framework                                    │
│    ├─ コンポーネントツリー生成                             │
│    ├─ セッションステート管理                               │
│    └─ Widget ID割り当て                                  │
└─────────────────────────────────────────────────────────┘
```

### PythonコードとHTMLの紐付けメカニズム

Streamlitは以下のメカニズムでPythonコードとブラウザのHTMLを紐付けます：

1. **Widget ID（一意識別子）**
   ```
   Pythonコードの位置 + ウィジェットタイプ + パラメータ
       ↓ ハッシュ化
   Widget ID (例: "btn_18f3a2b9e4c5")
   ```

2. **WebSocket通信（双方向メッセージング）**
   ```
   Browser → Server: "このWidget IDがクリックされた"
   Server → Browser: "この位置にこのコンポーネントを表示して"
   ```

3. **コンポーネントツリー（UIの構造化表現）**
   ```python
   # Pythonコード
   st.header("見出し")
   st.button("ボタン")

   # ↓ コンポーネントツリー（JSON）
   [
     {"type": "header", "text": "見出し"},
     {"type": "button", "id": "btn_xxx", "label": "ボタン"}
   ]

   # ↓ HTML
   <h2>見出し</h2>
   <button data-widget-id="btn_xxx">ボタン</button>
   ```

---

## 作業1: プロジェクトセットアップ

### 1-1. 依存パッケージの定義

**ファイル：[pyproject.toml](pyproject.toml)**

```toml
[project]
name = "geniac04"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "streamlit>=1.50.0",      # ← UI框架
    "pandas>=2.3.2",           # ← データ処理
    "langchain>=0.3.27",       # ← RAG処理
    # ... その他の依存関係
]
```

**開発者がやること：**
- Streamlitをプロジェクトの依存関係に追加
- バージョンを指定（`>=1.50.0`）

### 1-2. パッケージのインストールとアプリ起動

**コマンド：**
```bash
uv run streamlit run src/gui.py
```

**開発者がやること：**
- `uv`（パッケージマネージャ）でStreamlitをインストール
- アプリケーションを起動

---

## 作業2: 画面デザインの定義

### 2-1. ページ全体のレイアウト設定

**ファイル：[src/gui.py:80-81](src/gui.py#L80-L81)**

```python
def main():
    st.set_page_config(layout="wide")  # ← ページレイアウトを「ワイド」に設定
    init_session_state()
    setup_sidebar()
    pg = st.navigation([page_1, page_2, page_99])
    pg.run()
```

**開発者が定義していること：**

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `layout` | `"wide"` | 画面幅いっぱいに表示（デフォルトは中央寄せ） |

**Streamlitが自動生成するCSS：**
```css
.main .block-container {
    max-width: 100%;  /* ワイドレイアウト */
    padding: 3rem 1rem;
}
```

### 2-2. サイドバーのデザイン

**ファイル：[src/gui.py:55-78](src/gui.py#L55-L78)**

```python
def setup_sidebar():
    """サイドバーにモデル選択機能を追加"""
    with st.sidebar:  # ← サイドバーコンテキスト
        st.header("⚙️ 設定")

        st.subheader("LLMモデル選択")
        selected_model = st.selectbox(
            "使用するGeminiモデル",
            cfg.gemini_models,
            index=...,
            help="生成タスクに使用するGeminiモデルを選択してください"
        )

        st.divider()  # ← 区切り線
        st.caption(f"現在のモデル: **{st.session_state.selected_model}**")
```

**開発者が定義している画面デザイン要素：**

| コード | UI要素 | デザイン効果 |
|--------|--------|-------------|
| `with st.sidebar:` | サイドバー領域 | 左側に固定サイドバーを作成 |
| `st.header("⚙️ 設定")` | 大見出し | サイドバーのタイトル |
| `st.subheader(...)` | 中見出し | セクションの区切り |
| `st.selectbox(...)` | ドロップダウン | モデル選択UI |
| `st.divider()` | 水平線 | 視覚的な区切り |
| `st.caption(...)` | 小さいテキスト | 補足情報の表示 |

**Streamlitが自動生成するHTML：**
```html
<section class="st-emotion-cache-sidebar">
  <div class="st-emotion-cache-sidebar-content">
    <h2>⚙️ 設定</h2>
    <h3>LLMモデル選択</h3>
    <div class="stSelectbox">
      <select>...</select>
    </div>
    <hr class="st-divider" />
    <p class="st-caption">現在のモデル: <strong>gemini-1.5-flash</strong></p>
  </div>
</section>
```

### 2-3. メインページの階層構造デザイン

**ファイル：[src/ui/gui/page1.py:32-54](src/ui/gui/page1.py#L32-L54)**

```python
def page_1():
    # レベル1：ページタイトル
    st.title("GENIAC-PRIZE prototype v1.0")
    st.write("1. から 4. までを順番に実行してください。")

    # レベル2：セクション1
    st.header("1. 任意の出願を読み込む")
    step1()

    # レベル2：セクション2
    st.header("2. 情報探索 + 一致箇所表示")
    step2()

    # レベル2：セクション3
    st.header("3. 一致箇所表示")
    step3()

    # レベル2：セクション4
    st.header("4. 判断根拠出力")
    step4()

    # レベル3：その他セクション
    st.subheader("その他")
    step99()
```

**開発者が定義している階層構造：**

```
階層レベル1: st.title()     → <h1> (最大の見出し)
階層レベル2: st.header()    → <h2> (セクション見出し)
階層レベル3: st.subheader() → <h3> (サブセクション見出し)
階層レベル4: st.write()     → <p>  (本文テキスト)
階層レベル5: st.markdown("##### ...") → <h5> (小見出し)
```

### 2-4. 各セクション内のUI要素デザイン

#### セクション1：ファイルアップローダー

**ファイル：[src/ui/gui/page1.py:57-68](src/ui/gui/page1.py#L57-L68)**

```python
def step1():
    file_content = ""
    # ファイルアップローダーのデザイン
    uploaded_file = st.file_uploader(
        "1. XML形式の出願を１件アップロードしてください。",  # ← ラベル
        type=["xml", "txt"]  # ← 許可する拡張子
    )

    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        # テキストエリアのデザイン
        st.text_area(
            "ファイルの中身:",
            file_content,
            height=200  # ← 高さ指定
        )

        # 成功メッセージのデザイン
        st.success("ファイルがアップロードされました。検索結果や画面表示を初期化しました。")
```

**開発者が定義しているUI要素：**

| コード | UI要素 | デザインパラメータ |
|--------|--------|------------------|
| `st.file_uploader(...)` | ファイル選択ボタン | `type=["xml", "txt"]` で拡張子制限 |
| `st.text_area(...)` | 複数行テキスト表示 | `height=200` で高さ指定 |
| `st.success(...)` | 成功メッセージボックス | 緑色の背景で表示 |

#### セクション2：検索ボタンとデータ表示

**ファイル：[src/ui/gui/page1.py:71-79](src/ui/gui/page1.py#L71-L79)**

```python
def step2():
    # 説明文のデザイン
    st.write("出願の公開番号（query_id）、出願に紐づく公知例の公開番号（knowledge_id）、公知例の一致箇所（retrieved_chunk）を表示します。")

    # 検索ボタンのデザイン
    if st.button("検索", type="primary"):
        query: Patent = st.session_state.loader.run(QUERY_PATH)
        st.session_state.query = query
        st.session_state.df_retrieved = retrieve(st.session_state.retriever, query)

    # データフレーム表示のデザイン
    if not st.session_state.df_retrieved.empty:
        st.dataframe(
            st.session_state.df_retrieved[["query_id", "knowledge_id", "retrieved_chunk"]]
        )
```

**開発者が定義しているUI要素：**

| コード | UI要素 | デザインパラメータ |
|--------|--------|------------------|
| `st.button("検索", type="primary")` | プライマリボタン | 青色、大きめ、目立つデザイン |
| `st.dataframe(...)` | データテーブル | ソート可能、スクロール可能 |

**Streamlitが自動生成するHTML（検索ボタン）：**
```html
<div class="stButton">
  <button
    kind="primary"
    data-widget-id="button-page1.py:73-検索-xxx"
    type="button"
    class="st-emotion-cache-19rxjzo"
  >
    <div><span>検索</span></div>
  </button>
</div>
```

#### セクション3：ハイライト表示

**ファイル：[src/ui/gui/page1.py:82-95](src/ui/gui/page1.py#L82-L95)**

```python
def step3():
    st.write(f"一致箇所をハイライトし、その前後{MAX_CHAR}文字まで含めて表示します。")

    # 表示ボタンのデザイン
    if st.button("表示", type="primary"):
        st.session_state.matched_chunk_markdowns = []
        for i in range(n_chunk):
            markdown_text = create_matched_md(i, st.session_state.loader, MAX_CHAR)
            st.session_state.matched_chunk_markdowns.append(markdown_text)

    # ハイライト表示のデザイン
    if st.session_state.matched_chunk_markdowns:
        for i, md in enumerate(st.session_state.matched_chunk_markdowns):
            st.markdown(f"##### 一致箇所 {i + 1}/{n_chunk}")
            st.markdown(md, unsafe_allow_html=True)  # ← HTMLタグを許可
```

**開発者が定義しているデザイン要素：**

| コード | UI要素 | デザイン効果 |
|--------|--------|-------------|
| `st.markdown("##### ...")` | h5見出し | 小見出しで番号表示 |
| `st.markdown(md, unsafe_allow_html=True)` | HTML埋め込み | カスタムスタイルを適用 |

#### セクション4：プログレスバーとコードブロック

**ファイル：[src/ui/gui/page1.py:98-117](src/ui/gui/page1.py#L98-L117)**

```python
def step4():
    n_chunk = st.session_state.n_chunk

    # 生成ボタンのデザイン
    if st.button("生成", type="primary"):
        st.session_state.reasons = []

        # プログレスバーのデザイン
        status_text = st.empty()  # ← 動的に更新可能な領域
        progress = st.progress(0)  # ← プログレスバー（初期値0%）

        for i in range(n_chunk):
            status_text.text(f"{i + 1} / {n_chunk} 件目を生成中です...")
            reason = st.session_state.generator.generate(
                st.session_state.query,
                st.session_state.retrieved_docs[i]
            )
            st.session_state.reasons.append(reason)
            progress.progress((i + 1) / n_chunk)  # ← 進捗更新

        status_text.text("生成が完了しました。")

    # コードブロックのデザイン
    if st.session_state.reasons:
        for i, reason in enumerate(st.session_state.reasons):
            st.markdown(f"##### 判断根拠 {i + 1} / {n_chunk}")
            st.code(reason, language="markdown")  # ← コードブロック
```

**開発者が定義しているUI要素：**

| コード | UI要素 | デザイン効果 |
|--------|--------|-------------|
| `st.empty()` | プレースホルダー | 動的に内容を更新可能な領域 |
| `st.progress(0)` | プログレスバー | 0〜1の範囲で進捗表示 |
| `status_text.text(...)` | 動的テキスト | リアルタイムで状態を更新 |
| `st.code(reason, language="markdown")` | コードブロック | シンタックスハイライト付き表示 |

---

## 作業3: ボタンの振る舞いの定義

### 3-1. 検索ボタンの振る舞い全体

**ファイル：[src/ui/gui/page1.py:71-79](src/ui/gui/page1.py#L71-L79)**

```python
def step2():
    st.write("出願の公開番号（query_id）、...")

    # ========== ボタンの振る舞い定義 ==========
    if st.button("検索", type="primary"):  # ← 条件：ボタンがクリックされた
        # 処理1: データ読み込み
        query: Patent = st.session_state.loader.run(QUERY_PATH)

        # 処理2: セッションステート更新（上書き1）
        st.session_state.query = query

        # 処理3: 検索実行とセッションステート更新（上書き2）
        st.session_state.df_retrieved = retrieve(st.session_state.retriever, query)

    # ========== 結果表示の条件分岐 ==========
    if not st.session_state.df_retrieved.empty:
        st.dataframe(st.session_state.df_retrieved[["query_id", "knowledge_id", "retrieved_chunk"]])
```

### ボタンクリック時の完全な動作フロー

```
【時刻T0】ユーザーがブラウザで「検索」ボタンをクリック
  ↓
  HTML: <button>のclickイベント発火
  ↓
  JavaScript: Streamlitクライアントがイベントをキャプチャ
  ↓
  WebSocket: サーバーへメッセージ送信
  {
    "type": "rerun_script",
    "widget_states": {
      "button_key_xxx": { "clicked": true }
    }
  }
  ↓
【時刻T1】Streamlitサーバーがメッセージ受信
  ↓
  Pythonスクリプトの再実行開始
  main() → init_session_state() → page_1() → step2()
  ↓
【時刻T2】step2()内の if st.button("検索", type="primary") が True と評価
  ↓
【時刻T3】Override ① 実行
  OLD: st.session_state.query = Patent(A)
  NEW: st.session_state.query = Patent(B)  ← メモリ上で参照を切り替え
  ↓
【時刻T4】retrieve()関数呼び出し
  ├─ retriever.retrieve(query) でベクトル検索実行
  ├─ Override ②-A 実行
  │   OLD: st.session_state.retrieved_docs = [Doc1, Doc2]
  │   NEW: st.session_state.retrieved_docs = [Doc3, Doc4, Doc5]
  └─ DataFrame生成して return
  ↓
【時刻T5】Override ②-B 実行
  OLD: st.session_state.df_retrieved = DataFrame(2 rows)
  NEW: st.session_state.df_retrieved = DataFrame(3 rows)
  ↓
【時刻T6】if not st.session_state.df_retrieved.empty: が True と評価
  ↓
  st.dataframe(...) で新しい検索結果を表示
  ↓
【時刻T7】スクリプト実行完了、HTMLレスポンス生成
  ↓
【時刻T8】ブラウザが新しいHTMLを受信して画面更新
  ↓
  ユーザーに新しい検索結果が表示される ✓
```

### 開発者が定義している振る舞い

#### 振る舞い1：条件判定（クリック検知）

```python
if st.button("検索", type="primary"):
```

**動作フロー：**
```
ユーザーがクリック → Streamlitが True を返す → if文の中身を実行
クリックしていない → Streamlitが False を返す → if文をスキップ
```

#### 振る舞い2：データ読み込み

```python
query: Patent = st.session_state.loader.run(QUERY_PATH)
```

**動作フロー：**
```
QUERY_PATH (ファイルパス)
    ↓ st.session_state.loader.run()
Patent オブジェクト（パース済みデータ）
    ↓
query 変数に格納
```

#### 振る舞い3：状態の上書き（Override）

```python
st.session_state.query = query
st.session_state.df_retrieved = retrieve(...)
```

**動作フロー：**
```
【Before】
st.session_state.query = Patent(前回のデータ) または None
st.session_state.df_retrieved = DataFrame(前回の結果) または 空

【After】
st.session_state.query = Patent(今回のデータ)  ← 上書き
st.session_state.df_retrieved = DataFrame(今回の結果)  ← 上書き
```

### 3-2. 表示ボタンの振る舞い

**ファイル：[src/ui/gui/page1.py:82-95](src/ui/gui/page1.py#L82-L95)**

```python
def step3():
    st.write(f"一致箇所をハイライトし、その前後{MAX_CHAR}文字まで含めて表示します。")

    n_chunk = len(st.session_state.df_retrieved)
    st.session_state.n_chunk = n_chunk

    # ========== 表示ボタンの振る舞い ==========
    if st.button("表示", type="primary"):
        # 処理1: 既存データをクリア
        st.session_state.matched_chunk_markdowns = []

        # 処理2: ループで各チャンクを処理
        for i in range(n_chunk):
            markdown_text = create_matched_md(i, st.session_state.loader, MAX_CHAR)
            st.session_state.matched_chunk_markdowns.append(markdown_text)

    # ========== 表示結果の条件分岐 ==========
    if st.session_state.matched_chunk_markdowns:
        for i, md in enumerate(st.session_state.matched_chunk_markdowns):
            st.markdown(f"##### 一致箇所 {i + 1}/{n_chunk}")
            st.markdown(md, unsafe_allow_html=True)
```

**開発者が定義している振る舞い：**

| ステップ | コード | 振る舞いの説明 |
|---------|--------|---------------|
| 1. データクリア | `st.session_state.matched_chunk_markdowns = []` | 前回の表示内容を削除 |
| 2. ループ処理 | `for i in range(n_chunk):` | 検索結果の数だけ繰り返し |
| 3. マークダウン生成 | `create_matched_md(...)` | ハイライト付きテキストを生成 |
| 4. リストに追加 | `.append(markdown_text)` | 生成したテキストを保存 |
| 5. 条件表示 | `if st.session_state.matched_chunk_markdowns:` | データがあれば表示 |

### 3-3. 生成ボタンの振る舞い（プログレス付き）

**ファイル：[src/ui/gui/page1.py:98-117](src/ui/gui/page1.py#L98-L117)**

```python
def step4():
    n_chunk = st.session_state.n_chunk

    # ========== 生成ボタンの振る舞い ==========
    if st.button("生成", type="primary"):
        # 処理1: 既存データをクリア
        st.session_state.reasons = []

        # 処理2: プログレス表示用のUI要素作成
        status_text = st.empty()
        progress = st.progress(0)

        # 処理3: ループで各チャンクを処理
        for i in range(n_chunk):
            # 処理3-1: 進捗状況を表示（動的更新）
            status_text.text(f"{i + 1} / {n_chunk} 件目を生成中です...")

            # 処理3-2: LLMで判断根拠を生成
            reason = st.session_state.generator.generate(
                st.session_state.query,
                st.session_state.retrieved_docs[i]
            )

            # 処理3-3: 結果を保存
            st.session_state.reasons.append(reason)

            # 処理3-4: プログレスバーを更新
            progress.progress((i + 1) / n_chunk)

        # 処理4: 完了メッセージ表示
        status_text.text("生成が完了しました。")

    # ========== 生成結果の表示 ==========
    if st.session_state.reasons:
        for i, reason in enumerate(st.session_state.reasons):
            st.markdown(f"##### 判断根拠 {i + 1} / {n_chunk}")
            st.code(reason, language="markdown")
```

**開発者が定義している振る舞い（特殊な動的更新）：**

| ステップ | コード | 振る舞いの説明 |
|---------|--------|---------------|
| 1. プレースホルダー作成 | `status_text = st.empty()` | 後で更新可能な空領域を確保 |
| 2. プログレスバー作成 | `progress = st.progress(0)` | 0%の状態で表示 |
| 3. 動的テキスト更新 | `status_text.text(f"{i+1} / {n_chunk} 件目...")` | **同じ領域のテキストを上書き** |
| 4. プログレス更新 | `progress.progress((i+1) / n_chunk)` | **プログレスバーを更新** |

**リアルタイム更新の仕組み：**

```python
status_text = st.empty()  # ① 空の領域を確保

for i in range(5):
    status_text.text(f"{i+1} 件目")  # ② 同じ領域を何度も上書き
    # ループごとに画面が更新される！
```

**動作イメージ：**
```
ループ1: "1 件目を生成中です..."  [████░░░░░░] 20%
ループ2: "2 件目を生成中です..."  [████████░░] 40%
ループ3: "3 件目を生成中です..."  [████████████] 60%
...
```

### 3-4. リセットボタンの振る舞い

**ファイル：[src/ui/gui/page1.py:120-124](src/ui/gui/page1.py#L120-L124)**

```python
def step99():
    st.write("次の出願に対しても同様に、1. から順番に実行してください。")

    # ========== リセットボタンの振る舞い ==========
    if st.button("リセット"):
        # 処理1: セッションステートをリセット
        reset_session_state()

        # 処理2: 成功メッセージ表示
        st.success("クエリや検索結果の履歴をリセットしました。")
```

**リセット処理の詳細（[src/ui/gui/page1.py:24-29](src/ui/gui/page1.py#L24-L29)）：**

```python
def reset_session_state():
    st.session_state.df_retrieved = pd.DataFrame()  # 空のDataFrame
    st.session_state.matched_chunk_markdowns = []   # 空のリスト
    st.session_state.reasons = []                   # 空のリスト
    st.session_state.query = None                   # None
    st.session_state.retrieved_docs = []            # 空のリスト
```

---

## 作業4: データフローの定義

### 4-1. セッションステートの初期化

**ファイル：[src/gui.py:27-52](src/gui.py#L27-L52)**

```python
def init_session_state():
    # ========== 不変データ（一度作成したら変わらない） ==========
    if "loader" not in st.session_state:
        st.session_state.loader = CommonLoader()

    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(knowledge_dir=KNOWLEDGE_DIR)

    if "generator" not in st.session_state:
        st.session_state.generator = Generator()

    # ========== 可変データ（ボタン操作で変わる） ==========
    if "df_retrieved" not in st.session_state:
        st.session_state.df_retrieved = pd.DataFrame()

    if "matched_chunk_markdowns" not in st.session_state:
        st.session_state.matched_chunk_markdowns = []

    if "reasons" not in st.session_state:
        st.session_state.reasons = []

    if "query" not in st.session_state:
        st.session_state.query = None

    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = []

    if "file_id" not in st.session_state:
        st.session_state.file_id = "no_file_yet"

    if "n_chunk" not in st.session_state:
        st.session_state.n_chunk = 0

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = cfg.gemini_llm_name
```

### セッションステートの構造

```
【不変データ】- アプリ起動時に1回だけ作成
├─ loader      : XMLファイル読み込み用
├─ retriever   : ベクトル検索用
└─ generator   : LLM生成用

【可変データ】- ボタン操作で上書きされる
├─ query       : アップロードされた出願データ
├─ df_retrieved: 検索結果（DataFrame形式）
├─ retrieved_docs: 検索結果（Document配列）
├─ matched_chunk_markdowns: ハイライト表示用テキスト
├─ reasons     : LLM生成の判断根拠
├─ n_chunk     : 検索結果の件数
├─ file_id     : アップロードされたファイルのID
└─ selected_model: 選択されたLLMモデル
```

### 4-2. データフローの全体像

```
【ステップ1】ファイルアップロード (step1)
    ↓
uploaded_file → QUERY_PATH (ファイル保存)
                st.session_state.file_id = uploaded_file.file_id

【ステップ2】検索実行 (step2)
    ↓
QUERY_PATH → loader.run() → Patent オブジェクト
                            ↓
                    st.session_state.query = query
                            ↓
                    retriever.retrieve(query) → Document[]
                            ↓
                    st.session_state.retrieved_docs = [...]
                    st.session_state.df_retrieved = DataFrame(...)

【ステップ3】一致箇所表示 (step3)
    ↓
st.session_state.df_retrieved → create_matched_md() → Markdown文字列
                                                      ↓
                            st.session_state.matched_chunk_markdowns = [...]

【ステップ4】判断根拠生成 (step4)
    ↓
st.session_state.query + retrieved_docs → generator.generate() → 判断根拠テキスト
                                                                 ↓
                                        st.session_state.reasons = [...]
```

---

## 作業5: ビジネスロジックとの連携

### 5-1. 検索処理との連携

**ファイル：[src/ui/gui/utils.py:14-40](src/ui/gui/utils.py#L14-L40)**

```python
def retrieve(retriever: Retriever, query: Patent) -> pd.DataFrame:
    """検索を実行して、検索結果を返す"""
    # ========== ビジネスロジック呼び出し ==========
    retrieved_docs: list[Document] = retriever.retrieve(query)

    # ========== GUI状態管理 ==========
    st.session_state.retrieved_docs = retrieved_docs

    # ========== データ変換（Business → UI） ==========
    query_ids: list[str] = []
    knowledge_ids: list[str] = []
    retrieved_paths: list[str] = []
    retrieved_chunks: list[str] = []

    for doc in retrieved_docs:
        query_ids.append(query.publication.doc_number)
        knowledge_ids.append(doc.metadata["publication_number"])
        retrieved_paths.append(doc.metadata["path"])
        retrieved_chunks.append(doc.page_content)

    # ========== UI用のDataFrame作成 ==========
    df = pd.DataFrame({
        "query_id": query_ids,
        "knowledge_id": knowledge_ids,
        "retrieved_path": retrieved_paths,
        "retrieved_chunk": retrieved_chunks,
    })
    return df
```

**開発者が定義している連携処理：**

| 処理 | 説明 |
|------|------|
| ビジネスロジック呼び出し | `retriever.retrieve(query)` でベクトル検索を実行 |
| セッション保存 | `st.session_state.retrieved_docs = ...` で次のステップ用に保存 |
| データ変換 | `Document[]` → `DataFrame` に変換してUI表示用に整形 |

### 5-2. ハイライト生成との連携

**ファイル：[src/ui/gui/utils.py:53-76](src/ui/gui/utils.py#L53-L76)**

```python
def create_matched_md(index: int, xml_loader: CommonLoader, MAX_CHAR: int) -> str:
    """一致箇所とその前後MAX_CHAR文字を含めMarkdownテキストを作成する。"""

    # ========== GUI状態から取得 ==========
    chunk: str = st.session_state.df_retrieved["retrieved_chunk"].iloc[index]
    path: str = st.session_state.df_retrieved["retrieved_path"].iloc[index]

    # ========== ビジネスロジック呼び出し ==========
    knowledge: Patent = xml_loader.run(Path(path))
    knowledge_str: str = knowledge.to_str()

    # ========== データ処理 ==========
    normalized_chunk = _normalize_text(chunk)
    normalized_knowledge = _normalize_text(knowledge_str)

    parts: list[str] = normalized_knowledge.split(normalized_chunk)
    first_part: str = parts[0]
    second_part: str = parts[1] if len(parts) > 1 else ""

    # ========== HTML生成（カスタムスタイル） ==========
    markdown_text = f"""
        {first_part[-MAX_CHAR:]}
        <span style="background-color: yellow; color: black; padding: 2px 4px; border-radius: 3px;">
            {normalized_chunk}
        </span>
        {second_part[:MAX_CHAR]}
    """
    return markdown_text
```

---

## 作業6: スタイルとカスタマイズ

### 6-1. インラインCSSの定義

**ファイル：[src/ui/gui/utils.py:71-75](src/ui/gui/utils.py#L71-L75)**

```python
markdown_text = f"""
    {first_part[-MAX_CHAR:]}
    <span style="background-color: yellow; color: black; padding: 2px 4px; border-radius: 3px;">
        {normalized_chunk}
    </span>
    {second_part[:MAX_CHAR]}
"""
```

**開発者が定義しているカスタムスタイル：**

| CSSプロパティ | 値 | 効果 |
|--------------|-----|------|
| `background-color` | `yellow` | 黄色の背景 |
| `color` | `black` | 黒い文字 |
| `padding` | `2px 4px` | 内側の余白 |
| `border-radius` | `3px` | 角を丸く |

### 6-2. unsafe_allow_htmlの使用

**ファイル：[src/ui/gui/page1.py:95](src/ui/gui/page1.py#L95)**

```python
st.markdown(md, unsafe_allow_html=True)
```

**開発者が定義していること：**
- `unsafe_allow_html=True` でHTMLタグとインラインCSSを許可
- これにより、Streamlitのデフォルトスタイルを上書き可能

---

## 作業7: アプリケーションの実行

### 7-1. エントリーポイントの定義

**ファイル：[src/gui.py:80-89](src/gui.py#L80-L89)**

```python
def main():
    st.set_page_config(layout="wide")
    init_session_state()
    setup_sidebar()
    pg = st.navigation([page_1, page_2, page_99])
    pg.run()

if __name__ == "__main__":
    main()
```

**開発者が定義している実行順序：**
1. ページ設定（`st.set_page_config`）
2. セッション初期化（`init_session_state`）
3. サイドバー設定（`setup_sidebar`）
4. ページナビゲーション定義（`st.navigation`）
5. アプリ実行（`pg.run()`）

### 7-2. アプリの起動方法

**コマンド：**
```bash
uv run streamlit run src/gui.py
```

**開発者が実施すること：**
- ターミナルで上記コマンドを実行
- Streamlitサーバーが起動（デフォルト: http://localhost:8501）
- ブラウザで自動的に開く

---

## 開発チェックリスト

### ✅ 環境構築
- [ ] `pyproject.toml` に `streamlit>=1.50.0` を追加
- [ ] パッケージをインストール（`uv sync`）

### ✅ 画面デザイン
- [ ] ページレイアウト設定（`st.set_page_config(layout="wide")`）
- [ ] サイドバーデザイン（`with st.sidebar:`）
- [ ] 階層構造定義（`st.title`, `st.header`, `st.subheader`）
- [ ] UI要素配置（`st.button`, `st.dataframe`, `st.file_uploader`など）
- [ ] カスタムスタイル定義（インラインCSS、`unsafe_allow_html=True`）
- [ ] プログレスバー（`st.progress`, `st.empty`）

### ✅ ボタンの振る舞い
- [ ] クリック検知（`if st.button(...)`）
- [ ] データ読み込み処理
- [ ] セッションステート更新（上書き）
- [ ] 条件分岐による表示制御
- [ ] ループ処理と動的更新
- [ ] リセット処理

### ✅ データフロー
- [ ] セッションステート初期化（`init_session_state`）
- [ ] 不変データの定義（loader, retriever, generator）
- [ ] 可変データの定義（query, df_retrieved, reasonsなど）
- [ ] データの流れ定義（ファイル→検索→表示→生成）

### ✅ ビジネスロジック連携
- [ ] 検索処理呼び出し（`retriever.retrieve`）
- [ ] データ変換（Document → DataFrame）
- [ ] LLM生成呼び出し（`generator.generate`）
- [ ] ファイル読み込み（`loader.run`）

### ✅ スタイルとカスタマイズ
- [ ] インラインCSSの定義
- [ ] `unsafe_allow_html=True` の使用
- [ ] カスタムHTMLタグの埋め込み

### ✅ 実行
- [ ] エントリーポイント定義（`if __name__ == "__main__"`）
- [ ] アプリ起動（`streamlit run src/gui.py`）

---

## 重要なポイント

### Streamlitが自動でやってくれること

開発者は以下を書く必要がありません：

❌ HTMLタグの記述（`<div>`, `<button>`など）
❌ CSSスタイルの記述（`.css`ファイル）
❌ JavaScriptイベントハンドラの記述
❌ WebSocket通信の実装
❌ ボタンIDの生成と管理
❌ 画面の再レンダリング処理
❌ セッション管理の低レベル実装

### 開発者が書くのはPythonコードだけ！

```python
# これだけで完全に動作するUIが生成される
st.button("検索", type="primary")
st.dataframe(df)
st.progress(0.5)
```

### Streamlitの再実行モデル

- **ボタンをクリックするたびにスクリプト全体が再実行される**
- セッションステートだけが保持される
- ローカル変数は毎回初期化される

### Overrideの特性

```python
# これは「追加」ではなく「上書き」
st.session_state.df_retrieved = new_dataframe  # 古いデータは失われる

# 追加したい場合は明示的に連結が必要
# st.session_state.df_retrieved = pd.concat([old_df, new_df])
```

---

## 参考リンク

- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [プロジェクトREADME](README.md)
- [セットアップガイド](SETUP_GUIDE.md)

---

**最終更新日：2025-11-12**
