# 視覚検査アプリ

このアプリは、画像をアップロードして異常検知を行うStreamlitアプリケーションです。

## セットアップ

1. 必要なパッケージをインストール:

```bash
pip install -r requirements.txt
```

2. 環境変数を設定:

```bash
# env_templateファイルを.envにコピーして設定
cp env_template .env
```

`.env`ファイルを編集して、実際の値を設定してください:

```
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_access_token
SERVING_ENDPOINT_NAME=your_endpoint_name
BASE_URL=DATABRICKS_HOST/serving-endpoints
```

**注意**: Databricks認証情報が設定されていない場合、アプリはデモモードで動作し、サンプルの推論結果を表示します。

## アプリの実行

```bash
streamlit run app.py
```

## 使用方法

1. ブラウザでアプリが開きます
2. 「画像を選択してください」から画像ファイル（JPG, JPEG, PNG）をアップロード
3. アップロードされた画像が表示されます
4. 「推論を実行」ボタンをクリック
5. 推論結果が表示されます

## ファイル構成

- `app.py`: メインのStreamlitアプリケーション
- `func.py`: 画像処理と推論を行う関数群
- `requirements.txt`: 必要なPythonパッケージ
- `app.yaml`: アプリケーション設定ファイル

## トラブルシューティング

### Databricks認証エラーが発生する場合

1. `databricks-sdk`がインストールされていることを確認:

```bash
pip install databricks-sdk
```

2. 環境変数が正しく設定されていることを確認:

```bash
echo $DATABRICKS_HOST
echo $DATABRICKS_TOKEN
```

3. Databricks認証情報が設定されていない場合、アプリは自動的にデモモードで動作します。
