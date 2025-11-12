# Page 3 実装ガイド

## 概要
このドキュメントは、GUIアプリケーションに新しいページ（page3）を追加した作業手順をまとめたものです。

## 実装日
2025-11-12

## 作業内容

### 1. 既存のGUI構造の調査
まず、既存のコードベースを調査して、ページの実装方法を理解しました。

**調査したファイル:**
- `/home/sonozuka/staging/patent_rag/src/gui.py` - メインのGUIファイル
- `/home/sonozuka/staging/patent_rag/src/ui/gui/page1.py` - 既存のページ1
- `/home/sonozuka/staging/patent_rag/src/ui/gui/page2.py` - 既存のページ2
- `/home/sonozuka/staging/patent_rag/src/ui/gui/page99.py` - 既存のページ99

**発見した構造:**
- 各ページは `ui/gui/` ディレクトリ内に独立したPythonファイルとして存在
- 各ページファイルは `page_X()` という関数を定義
- `gui.py` でページをインポートし、`st.navigation()` に渡してナビゲーションを構築

### 2. page3.py の作成
新しいページファイルを作成しました。

**ファイルパス:** `/home/sonozuka/staging/patent_rag/src/ui/gui/page3.py`

**実装内容:**
```python
import streamlit as st


def page_3():
    st.write("page3です")

    # Hello!ボタンを表示
    if st.button("Hello!"):
        print("hello")
        st.success("ターミナルに 'hello' を出力しました！")
```

**機能:**
- ページタイトルとして "page3です" を表示
- "Hello!" ボタンを配置
- ボタンがクリックされたら:
  - ターミナルに `print("hello")` を実行
  - ユーザーに成功メッセージを表示

### 3. gui.py の更新
メインのGUIファイルにpage3を統合しました。

#### 3.1 インポートの追加
**変更箇所:** [gui.py:16-19](src/gui.py#L16-L19)

```python
from ui.gui.page1 import page_1
from ui.gui.page2 import page_2
from ui.gui.page3 import page_3  # 追加
from ui.gui.page99 import page_99
```

#### 3.2 ナビゲーションへの追加
**変更箇所:** [gui.py:85](src/gui.py#L85)

```python
pg = st.navigation([page_1, page_2, page_3, page_99])
```

**配置:** page_2とpage_99の間にpage_3を配置しました。

## 動作確認方法

### GUIアプリケーションの起動
```bash
cd /home/sonozuka/staging/patent_rag
streamlit run src/gui.py
```

### 確認項目
1. ブラウザでアプリケーションが開く
2. ナビゲーションメニューに "page_3" が表示される
3. page_3をクリックすると、"page3です" というテキストが表示される
4. "Hello!" ボタンが表示される
5. ボタンをクリックすると:
   - ターミナル（streamlitを起動したコンソール）に "hello" が出力される
   - 画面に成功メッセージが表示される

## ファイル構成

```
patent_rag/
├── src/
│   ├── gui.py                    # メインGUIファイル（更新）
│   └── ui/
│       └── gui/
│           ├── page1.py          # 既存
│           ├── page2.py          # 既存
│           ├── page3.py          # 新規作成
│           └── page99.py         # 既存
└── docs/
    └── page3_implementation_guide.md  # このドキュメント
```

## 技術詳細

### 使用しているフレームワーク
- **Streamlit**: Pythonのウェブアプリケーションフレームワーク
  - `st.write()`: テキストやマークダウンを表示
  - `st.button()`: ボタンウィジェットを作成
  - `st.success()`: 成功メッセージを表示
  - `st.navigation()`: マルチページナビゲーションを構築

### ページの動作原理
1. Streamlitは各ページを関数として管理
2. `st.navigation()` にページ関数のリストを渡すことで、ナビゲーションメニューを自動生成
3. ユーザーがページを選択すると、該当する関数が実行される
4. ボタンのクリックイベントは `st.button()` の戻り値で検出

### print()の動作
- `print("hello")` はPythonの標準出力に出力される
- Streamlitアプリケーションを起動しているターミナル/コンソールに表示される
- ブラウザには表示されない（ターミナルのみ）

## 今後の拡張案
- ボタンのスタイルをカスタマイズ
- クリック回数のカウンター機能を追加
- 異なるメッセージを表示するボタンを追加
- ログファイルへの出力機能を追加

## トラブルシューティング

### "hello"がターミナルに表示されない場合
- Streamlitを起動したターミナル/コンソールウィンドウを確認してください
- ブラウザのコンソールではなく、サーバー側のターミナルに出力されます

### ページが表示されない場合
- `gui.py` で正しくインポートされているか確認
- `st.navigation()` のリストに `page_3` が含まれているか確認
- ファイルパスが正しいか確認（`src/ui/gui/page3.py`）

## ボタン名の変更手順

### ボタン名を「Hello!」から「クエリー」に変更する方法

Page 3のボタン名を変更したい場合、以下の手順で実施します。

#### 手順1: page3.pyファイルを開く
```bash
# エディタで以下のファイルを開きます
vi /home/sonozuka/staging/patent_rag/src/ui/gui/page3.py
# または
code /home/sonozuka/staging/patent_rag/src/ui/gui/page3.py
```

#### 手順2: ボタンのラベルを変更
`st.button()` の引数を変更します。

**変更前:**
```python
def page_3():
    st.write("page3です")

    # Hello!ボタンを表示
    if st.button("Hello!"):
        print("hello")
        st.success("ターミナルに 'hello' を出力しました！")
```

**変更後:**
```python
def page_3():
    st.write("page3です")

    # クエリーボタンを表示
    if st.button("クエリー"):
        print("hello")
        st.success("ターミナルに 'hello' を出力しました！")
```

#### 手順3: 変更を保存
ファイルを保存します（`Ctrl+S` または `:wq`）

#### 手順4: Streamlitアプリケーションを再起動
変更を反映するため、以下の方法でアプリケーションを更新します：

**方法1: ブラウザで「Always rerun」を選択**
- ブラウザの右上に表示される「Source file changed」通知で「Always rerun」をクリック

**方法2: ターミナルでアプリケーションを再起動**
```bash
# Ctrl+C でアプリを停止
# 再度起動
streamlit run src/gui.py
```

#### 確認
ブラウザでPage 3を表示すると、ボタンのラベルが「クエリー」に変更されていることを確認できます。

### ボタン名変更の応用
同じ方法で、他の文字列にも変更できます：
- 「検索」
- 「実行」
- 「送信」
など、用途に応じたラベルに変更可能です。

### 変更履歴
- **2025-11-12 初回実装**: ボタン名を「Hello!」で作成
- **2025-11-12 更新**: ボタン名を「クエリー」に変更

## まとめ
Page 3の実装は以下の3つのステップで完了しました:
1. 既存のページ構造を調査・理解
2. 新しい `page3.py` ファイルを作成し、Hello!ボタンとprint機能を実装
3. `gui.py` を更新してナビゲーションに追加

その後、ボタン名を「Hello!」から「クエリー」に変更しました。

この実装により、ユーザーはGUIのナビゲーションメニューからPage 3にアクセスし、クエリーボタンをクリックしてターミナルに出力を確認できるようになりました。
