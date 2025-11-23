# Cloud Run 更新手順書（実践版）

- 2025年11月23日作成
- URLを固定したまま、アプリケーションのコードを更新する手順

## 概要

Cloud Runサービス **patent-rag-web** の公開URLを変更せず、アプリケーション（RAG・AI審査・UI など）の中身だけを継続的に更新するための実践手順。

**公開URL（固定）**:
```
https://patent-rag-web-453242904538.us-central1.run.app/
```

---

## 前提条件

### 必須ツール
- Google Cloud SDK (gcloud CLI) がインストール済み
- gcloud認証が完了していること
- プロジェクトへのアクセス権限があること

### 必要なAPI（有効化済みであること）
- Artifact Registry API
- Cloud Build API
- Cloud Run API
- Secret Manager API

### 環境変数ファイル (.env)

プロジェクトルート（`/home/sonozuka/staging/patent_rag/.env`）に以下の環境変数を設定：

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-proj-xxxxx

# Google Gemini API Key
GOOGLE_API_KEY=AIzaSyBxxxxxx

# BigQuery設定
GCP_PROJECT_ID=llmatch-471107
DATASET_ID=google_dataset
TABLE_ID=google_japan_patents
```

---

## 更新手順

### ステップ1: gcloud認証の確認

まず、gcloud CLIが正しく認証されているか確認します。

```bash
# 認証状態を確認
gcloud auth list

# 認証されていない場合は、以下を実行
gcloud auth login

# プロジェクトを設定
gcloud config set project llmatch-471107
```

**確認コマンド**:
```bash
gcloud config get-value project
# 出力: llmatch-471107
```

---

### ステップ2: 環境変数の設定

デプロイに必要な環境変数を設定します。

```bash
# プロジェクトディレクトリに移動
cd /home/sonozuka/staging/patent_rag

# .envファイルから基本的な環境変数をロード
set -a
source .env
set +a

# Cloud Run デプロイ用の追加環境変数を設定
export REGION="us-central1"
export GCP_PROJECT_ID="llmatch-471107"
export REPO="geniac"
export SERVICE_NAME="patent-rag-web"
export PROJECT_NUMBER="453242904538"
export IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO}/patent-rag:latest"
export SA_EMAIL="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
```

**環境変数の確認**:
```bash
echo "REGION: $REGION"
echo "GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "SERVICE_NAME: $SERVICE_NAME"
echo "IMAGE: $IMAGE"
echo "SA_EMAIL: $SA_EMAIL"
```

**期待される出力**:
```
REGION: us-central1
GCP_PROJECT_ID: llmatch-471107
SERVICE_NAME: patent-rag-web
IMAGE: us-central1-docker.pkg.dev/llmatch-471107/geniac/patent-rag:latest
SA_EMAIL: 453242904538-compute@developer.gserviceaccount.com
```

---

### ステップ3: コンテナイメージのビルド

Dockerfileを使用してコンテナイメージをビルドし、Artifact Registryにプッシュします。

```bash
# プロジェクトルートにいることを確認
cd /home/sonozuka/staging/patent_rag

# Cloud Buildでイメージをビルド（所要時間: 約5-10分）
gcloud builds submit \
  --tag ${IMAGE} \
  --region=${REGION} \
  --project="${GCP_PROJECT_ID}"
```

**実行内容**:
- Dockerfileに基づいてコンテナイメージをビルド
- Python 3.11環境のセットアップ
- uvを使用した依存関係のインストール
- Streamlitアプリケーションの準備
- Artifact Registryへのイメージプッシュ

**成功時の出力（最終行）**:
```
SUCCESS
```

---

### ステップ4: Cloud Runへのデプロイ

ビルドしたコンテナイメージをCloud Runにデプロイします。

```bash
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE} \
  --region ${REGION} \
  --project="${GCP_PROJECT_ID}" \
  --allow-unauthenticated \
  --service-account ${SA_EMAIL} \
  --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest \
  --set-env-vars GCP_PROJECT_ID=${GCP_PROJECT_ID} \
  --set-env-vars DATASET_ID=${DATASET_ID} \
  --set-env-vars TABLE_ID=${TABLE_ID} \
  --memory=2Gi \
  --cpu=2 \
  --timeout=900
```

**デプロイオプションの説明**:
- `--allow-unauthenticated`: 認証なしでアクセス可能
- `--service-account`: 実行時のサービスアカウント
- `--set-secrets`: Secret Managerからシークレットを注入
- `--set-env-vars`: 環境変数を設定
- `--memory=2Gi`: メモリを2GBに設定
- `--cpu=2`: CPUを2コアに設定
- `--timeout=900`: タイムアウトを900秒（15分）に設定

**成功時の出力**:
```
Deploying container to Cloud Run service [patent-rag-web] in project [llmatch-471107] region [us-central1]
Deploying...
Setting IAM Policy........................done
Creating Revision.........done
Routing traffic.....done
Done.
Service [patent-rag-web] revision [patent-rag-web-00002-cj5] has been deployed and is serving 100 percent of traffic.
Service URL: https://patent-rag-web-453242904538.us-central1.run.app
```

---

### ステップ5: デプロイの確認

デプロイが成功したら、以下を確認します。

#### 5-1. Cloud Runコンソールでの確認

Google Cloud Console → Cloud Run → patent-rag-web で以下を確認：
- ✅ ステータスが「準備完了（Ready）」
- ✅ 新しいRevisionが作成されている
- ✅ トラフィック割当が100%になっている

#### 5-2. ブラウザでの動作確認

以下のURLをブラウザで開き、アプリケーションが正しく動作するか確認：
```
https://patent-rag-web-453242904538.us-central1.run.app
```

#### 5-3. ログの確認

デプロイ後のログを確認：
```bash
gcloud run services logs read patent-rag-web \
  --region=us-central1 \
  --project=llmatch-471107 \
  --limit=50
```

---

## 簡易コマンド版（2ステップ）

環境変数が設定済みの場合、以下の2コマンドで更新可能：

### 1. ビルド
```bash
gcloud builds submit --tag ${IMAGE} --region=${REGION} --project="${GCP_PROJECT_ID}"
```

### 2. デプロイ
```bash
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE} \
  --region ${REGION} \
  --project="${GCP_PROJECT_ID}" \
  --allow-unauthenticated \
  --service-account ${SA_EMAIL} \
  --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest \
  --set-env-vars GCP_PROJECT_ID=${GCP_PROJECT_ID},DATASET_ID=${DATASET_ID},TABLE_ID=${TABLE_ID} \
  --memory=2Gi --cpu=2 --timeout=900
```

---

## トラブルシューティング

### 問題1: gcloud認証エラー

**エラーメッセージ**:
```
You do not currently have an active account selected.
```

**解決方法**:
```bash
gcloud auth login
gcloud config set project llmatch-471107
```

---

### 問題2: ビルドエラー

**エラー**: Cloud Build APIが無効

**解決方法**:
```bash
gcloud services enable cloudbuild.googleapis.com --project=llmatch-471107
```

---

### 問題3: デプロイエラー

**エラー**: Secret Manager APIが無効、またはシークレットが見つからない

**解決方法**:

1. Secret Manager APIを有効化：
```bash
gcloud services enable secretmanager.googleapis.com --project=llmatch-471107
```

2. シークレットを作成：
```bash
printf "%s" "${GOOGLE_API_KEY}" | gcloud secrets create GOOGLE_API_KEY \
  --data-file=- \
  --project="${GCP_PROJECT_ID}"

# サービスアカウントに権限を付与
gcloud secrets add-iam-policy-binding GOOGLE_API_KEY \
  --project=${GCP_PROJECT_ID} \
  --member=serviceAccount:${SA_EMAIL} \
  --role=roles/secretmanager.secretAccessor
```

---

### 問題4: Artifact Registryへのアクセスエラー

**エラー**: リポジトリが存在しない

**解決方法**:
```bash
# リポジトリを作成
gcloud artifacts repositories create ${REPO} \
  --project="${GCP_PROJECT_ID}" \
  --location="${REGION}" \
  --repository-format=docker \
  --description="Containers for patent-rag"

# IAM権限を付与
gcloud artifacts repositories add-iam-policy-binding ${REPO} \
  --project="${GCP_PROJECT_ID}" \
  --location="${REGION}" \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
```

---

## ロールバック手順

デプロイ後に問題が発生した場合、以前のRevisionに戻すことができます。

### Cloud Consoleでのロールバック

1. Cloud Run → patent-rag-web → 「リビジョン」タブ
2. 前のリビジョンを選択
3. 「トラフィックの管理」→ 100%に設定

### CLIでのロールバック

```bash
# リビジョン一覧を表示
gcloud run revisions list --service=patent-rag-web --region=us-central1

# 特定のリビジョンにトラフィックを100%割り当て
gcloud run services update-traffic patent-rag-web \
  --to-revisions=patent-rag-web-00001-xxx=100 \
  --region=us-central1
```

---

## 更新サイクルのベストプラクティス

### 1. ローカルでの開発とテスト
- コードの修正
- ローカル環境でのテスト（`streamlit run src/gui.py`）

### 2. 依存関係の更新
```bash
# 新しいパッケージを追加
uv add パッケージ名

# 依存関係を同期
uv sync
```

### 3. ビルドとデプロイ
- 上記の手順に従ってビルド・デプロイ

### 4. 動作確認
- URLにアクセスして動作確認
- ログの確認

### 5. 必要に応じてロールバック
- 問題があれば前のRevisionに戻す

---

## まとめ

Cloud Runの更新は以下の流れで行います：

1. **認証確認** → gcloud auth login
2. **環境変数設定** → export で必要な変数を設定
3. **ビルド** → gcloud builds submit
4. **デプロイ** → gcloud run deploy
5. **確認** → ブラウザとログで動作確認

この手順により、**URLを変更せずに**アプリケーションの中身だけを更新できます。

---

## 参考資料

- [Cloud Run ドキュメント](https://cloud.google.com/run/docs)
- [Cloud Build ドキュメント](https://cloud.google.com/build/docs)
- [Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs)