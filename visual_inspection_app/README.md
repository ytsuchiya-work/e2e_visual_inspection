# 視覚検査アプリ

このアプリは、画像をアップロードして異常検知を行うStreamlitアプリケーションです。

## セットアップ

1. 仮想環境のセットアップ

```bash
uv venv .venv
source .venv/bin/activate
```

2. 必要なパッケージをインストール:

```bash
uv pip install -r requirements.txt
```

3. 環境変数を設定:

`.env`ファイルを編集して、実際の値を設定してください:

```
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_access_token
SERVING_ENDPOINT_NAME=your_endpoint_name
BASE_URL=<DATABRICKS_HOST>/serving-endpoints
```

**注意**: Databricks認証情報が設定されていない場合、アプリはデモモードで動作し、サンプルの推論結果を表示します。

## アプリの実行

```bash
streamlit run app.py
```

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
