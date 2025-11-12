# デバッグガイド

このドキュメントは、Streamlitアプリケーションのデバッグ手順をまとめたものです。
VSCodeのデバッガを使用して、実際の動作に近い形でアプリケーションをデバッグできます。

## 目次

- [準備](#準備)
- [方法1: UI連携のデバッグ（推奨）](#方法1-ui連携のデバッグ推奨)
- [方法2: ロジック単体のデバッグ](#方法2-ロジック単体のデバッグ)
- [デバッグのコツ](#デバッグのコツ)
- [トラブルシューティング](#トラブルシューティング)

---

## 準備

以下のファイルが既に準備されています：

### 1. デバッグ設定ファイル

**ファイル**: `.vscode/launch.json`

以下の2つのデバッグ設定が含まれています：

- **Python デバッガー: 現在のファイル**
  - 通常のPythonファイルをデバッグする際に使用
  - `debug_retriever.py`、`debug_generator.py`などのスクリプト実行に使用

- **Python: Streamlit (gui.py)**
  - Streamlitアプリをデバッグモードで起動
  - UI操作を含む実際の動作をデバッグする際に使用

### 2. ロジック単体デバッグ用スクリプト

- **`debug_retriever.py`**: 検索ロジック（Retriever）のデバッグ用
- **`debug_generator.py`**: 判断根拠生成ロジック（Generator）のデバッグ用

---

## 方法1: UI連携のデバッグ（推奨）

**推奨理由**: 実際のユーザー操作と同じ流れでデバッグでき、`st.session_state`などのStreamlit固有の状態も確認できます。

### ステップ1: ブレークポイントを設定

1. VSCodeで `src/ui/gui/page1.py` を開く

2. デバッグしたい処理の行にブレークポイントを設定
   - 行番号の左側をクリックすると赤い点が表示されます

3. **推奨ブレークポイント箇所**

   #### 検索処理のデバッグ（Step 2）
   ```python
   # src/ui/gui/page1.py: 71-79行目
   def step2():
       st.write("出願の公開番号...")
       if st.button("検索", type="primary"):
           query: Patent = st.session_state.loader.run(QUERY_PATH)  # ←ここ（74行目）
           st.session_state.query = query
           st.session_state.df_retrieved = retrieve(st.session_state.retriever, query)  # ←または、ここ（76行目）
   ```

   #### 一致箇所表示のデバッグ（Step 3）
   ```python
   # src/ui/gui/page1.py: 82-95行目
   def step3():
       if st.button("表示", type="primary"):
           st.session_state.matched_chunk_markdowns = []
           for i in range(n_chunk):
               markdown_text = create_matched_md(i, st.session_state.loader, MAX_CHAR)  # ←ここ（89行目）
   ```

   #### 判断根拠生成のデバッグ（Step 4）
   ```python
   # src/ui/gui/page1.py: 98-117行目
   def step4():
       if st.button("生成", type="primary"):
           for i in range(n_chunk):
               reason = st.session_state.generator.generate(st.session_state.query, st.session_state.retrieved_docs[i])  # ←ここ（109行目）
   ```

> **重要**: `if st.button(...):`の行自体にはブレークポイントを置かないでください。
> ボタンがクリックされていないときも止まってしまいます。
> 必ず`if`ブロックの**内部**にある実際の処理呼び出し行に設定してください。

### ステップ2: デバッグを開始

1. VSCodeの左サイドバーで **「実行とデバッグ」** アイコンをクリック
   - アイコン: 再生ボタンと虫のマーク

2. 上部のドロップダウンから **「Python: Streamlit (gui.py)」** を選択

3. **F5キー** を押す（または緑の再生ボタンをクリック）

4. 統合ターミナルにStreamlitのログが表示され、ブラウザが自動で開きます
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   ```

### ステップ3: ブラウザでアプリを操作

1. ブラウザで開いたStreamlitアプリで以下を実行：

   #### 検索処理のデバッグの場合
   - **「1. 任意の出願を読み込む」** セクションでXMLファイルをアップロード
   - **「2. 情報探索」** セクションで **「検索」ボタン** をクリック

   #### 一致箇所表示のデバッグの場合
   - 上記に加えて、**「3. 一致箇所表示」** セクションで **「表示」ボタン** をクリック

   #### 判断根拠生成のデバッグの場合
   - 上記に加えて、**「4. 判断根拠出力」** セクションで **「生成」ボタン** をクリック

2. ボタンをクリックすると、**VSCodeでブレークポイントの行で実行が停止**します

### ステップ4: ステップ実行

デバッグツールバー（画面上部に表示）を使用：

| キー | 操作 | 説明 |
|------|------|------|
| **F10** | ステップオーバー | 次の行へ進む（関数呼び出しは実行するが中には入らない） |
| **F11** | ステップイン | 関数の中に入る |
| **Shift+F11** | ステップアウト | 現在の関数から抜ける |
| **F5** | 続行 | 次のブレークポイントまで実行 |
| **Shift+F5** | 停止 | デバッグを終了 |

### ステップ5: 変数の確認

左サイドバーの **「変数」** タブで、以下の値を確認できます：

- `query`: 読み込まれた特許データ
- `retriever`: 検索エンジンのインスタンス
- `df_retrieved`: 検索結果のDataFrame
- `st.session_state`: Streamlitのセッション状態

**ウォッチ機能**: 特定の変数を追跡したい場合
1. 左サイドバーの **「ウォッチ」** タブをクリック
2. **「+」** ボタンをクリックして変数名を入力（例: `query.publication.doc_number`）

### ステップ6: デバッグを終了

- **Shift+F5** でデバッグを停止
- または、デバッグツールバーの赤い四角ボタンをクリック

---

## 方法2: ロジック単体のデバッグ

**使用場面**: UIが関係ない検索ロジックや生成ロジック自体のバグを高速に潰したい場合

### 検索ロジック（Retriever）のデバッグ

#### ステップ1: ファイルを開く

VSCodeで `debug_retriever.py` を開く

#### ステップ2: ブレークポイントを設定

以下のいずれかの箇所にブレークポイントを設定：

- **`debug_retriever.py`の38行目**
  ```python
  retrieved_docs = retriever.retrieve(query_patent)  # ←ここ
  ```

- **`src/app/retriever.py`の93行目**（関数内部に入りたい場合）
  ```python
  def retrieve(self, query: str | Patent) -> list[Document]:
      # ←ここから内部をステップ実行
  ```

#### ステップ3: デバッグを開始

1. **F5キー** を押す
2. 「**Python デバッガー: 現在のファイル**」を選択
3. スクリプトが実行され、ブレークポイントで停止します

#### ステップ4: ステップ実行

F10/F11キーでステップ実行し、以下を確認：

- `query_patent`: 読み込まれたテスト用クエリ
- `retriever.chroma`: ベクトルストアの状態
- `retrieved_docs`: 検索結果

#### 出力例

デバッグコンソールに以下のような出力が表示されます：

```
============================================================
Retriever デバッグ開始
============================================================

[Step 1] Retrieverを初期化中...
✓ Retrieverの初期化完了

[Step 2] テストクエリを読み込み中: eval/knowledge/result_1/0/JP2010000001A/text.txt
✓ クエリ読み込み完了
  - 出願番号: JP2010000001A
  - 発明の名称: ...

[Step 3] 類似特許を検索中...
# ←ここでブレークポイントで停止
```

---

### 生成ロジック（Generator）のデバッグ

#### ステップ1: ファイルを開く

VSCodeで `debug_generator.py` を開く

#### ステップ2: ブレークポイントを設定

以下のいずれかの箇所にブレークポイントを設定：

- **`debug_generator.py`の52行目**
  ```python
  reason = generator.generate(query_patent, first_doc)  # ←ここ
  ```

- **`src/app/generator.py`の35行目**（関数内部に入りたい場合）
  ```python
  def generate(self, query: Patent, retrieved_doc: Document) -> str:
      # ←ここから内部をステップ実行
  ```

#### ステップ3-4: デバッグを開始・実行

検索ロジックと同様の手順で実行します。

#### 確認項目

- `query_content`: クエリ特許の文字列表現
- `retrieved_doc_content`: 検索結果の内容
- `prompt`: LLMに送信されるプロンプト
- `response`: LLMからの応答
- `reason`: 生成された判断根拠テキスト

---

## デバッグのコツ

### 1. 最初は方法1（UI連携）を推奨

- 実際のユーザー操作と同じ流れでデバッグできます
- `st.session_state`の状態も確認できます
- ファイルアップロードなどのUI操作も含めてテストできます

### 2. ロジックのバグが見つかったら方法2で深掘り

- UIの起動なしで高速にテストできます
- 特定の関数を集中的に確認できます
- テストデータを変更して複数パターンを試しやすい

### 3. ブレークポイントを複数設定

- 処理の流れを追いやすくなります
- 不要になったブレークポイントは右クリックで削除
- 条件付きブレークポイント：右クリック → 「条件付きブレークポイントの追加」

### 4. 変数ウォッチ機能を活用

- 左サイドバーの「ウォッチ」タブで変数名を追加
- 変数の変化を追跡できます
- 例: `query.publication.doc_number`、`len(retrieved_docs)`

### 5. デバッグコンソールを活用

- デバッグ中に「デバッグコンソール」タブで任意の式を評価できます
- 例: `query.to_str()`、`type(retriever)`

### 6. ログ出力との併用

- `print()`文を使った従来のデバッグと併用すると効果的です
- 統合ターミナルにログが表示されます

---

## トラブルシューティング

### ブレークポイントで止まらない

**原因1**: ブレークポイントを`if st.button(...):` の行自体に設定している

- **解決策**: `if`ブロックの**内部**にある処理呼び出し行に設定

**原因2**: ボタンをクリックしていない

- **解決策**: ブラウザでボタンをクリックしてから待つ

**原因3**: `justMyCode: true` の設定

- **解決策**: サードパーティライブラリ内部もデバッグしたい場合は、`.vscode/launch.json`の`justMyCode`を`false`に変更

### エラー: `ModuleNotFoundError`

**原因**: `PYTHONPATH`が正しく設定されていない

**解決策**: `.vscode/launch.json`に以下が含まれているか確認
```json
"env": {
    "PYTHONPATH": "${workspaceFolder}/src"
}
```

### Streamlitアプリが起動しない

**原因1**: ポート8501が既に使用されている

**解決策**: 既存のStreamlitプロセスを終了するか、別のポートを指定
```json
"args": [
    "run",
    "src/gui.py",
    "--server.port=8502"
]
```

**原因2**: 環境変数（API KEYなど）が読み込まれていない

**解決策**: `.env`ファイルが正しい場所にあるか確認

### デバッグが遅い

**原因**: ベクトルストアの初期化に時間がかかる

**解決策**:
1. 初回実行後、ベクトルストアが`persist_dir`に保存されるので2回目以降は高速
2. 方法2（ロジック単体）を使用して、必要な部分だけデバッグ

---

## 参考資料

### 主要ファイルとデバッグ対象

| ファイル | 内容 | デバッグ対象 |
|---------|------|-------------|
| `src/gui.py` | Streamlitアプリのエントリーポイント | セッション初期化、モデル選択 |
| `src/ui/gui/page1.py` | メインページのUI定義 | ファイルアップロード、ボタン処理 |
| `src/ui/gui/utils.py` | UI用のユーティリティ関数 | `retrieve()`, `create_matched_md()` |
| `src/app/retriever.py` | 検索エンジン | ベクトル検索、Chroma操作 |
| `src/app/generator.py` | 判断根拠生成器 | LLM API呼び出し、プロンプト生成 |
| `src/app/rag.py` | RAGパイプライン | Retriever + Generator の統合 |
| `src/infra/loader/common_loader.py` | XMLローダー | 特許XMLファイルの読み込み |
| `src/model/patent.py` | 特許データモデル | データ構造の定義 |

### よく使うブレークポイント箇所

| 箇所 | ファイル:行 | タイミング |
|------|-----------|-----------|
| XMLファイル読み込み | `page1.py:74` | 「検索」ボタンクリック時 |
| 検索実行 | `utils.py:23` | `retrieve()`関数呼び出し時 |
| ベクトル検索 | `retriever.py:107` | `retriever.invoke()`実行時 |
| 一致箇所作成 | `utils.py:64-73` | マークダウン生成時 |
| 判断根拠生成 | `generator.py:46` | LLM API呼び出し時 |

### VSCodeショートカット一覧

| 操作 | Windows/Linux | macOS |
|------|---------------|-------|
| デバッグ開始 | F5 | F5 |
| ステップオーバー | F10 | F10 |
| ステップイン | F11 | F11 |
| ステップアウト | Shift+F11 | Shift+F11 |
| 続行 | F5 | F5 |
| デバッグ停止 | Shift+F5 | Shift+F5 |
| ブレークポイント切替 | F9 | F9 |
| 実行とデバッグビュー表示 | Ctrl+Shift+D | Cmd+Shift+D |

---

## まとめ

このデバッグ環境を使用することで：

✅ Streamlitアプリを実際の動作に近い形でデバッグできます
✅ UI操作を含む複雑な処理フローを追跡できます
✅ ロジック単体の高速デバッグも可能です
✅ 変数の状態を逐一確認しながら開発できます

問題が発生した場合は、トラブルシューティングセクションを参照してください。

Happy Debugging! 🐛🔍
