# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # スケールとリアルタイムでの推論実行
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-5.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC モデルをレジストリにデプロイしました。レジストリはガバナンスとACLを提供し、下流のパイプライン開発を簡素化・加速します。
# MAGIC
# MAGIC 他のチームはモデル自体を気にせず、運用タスクやモデルサービングに集中でき、データサイエンティストは準備ができたときに新しいモデルをリリースできます。
# MAGIC
# MAGIC モデルの主な利用方法は2つです：
# MAGIC
# MAGIC - スケール用途：クラスター上でのバッチまたはストリーミング（Delta Live Tables含む）
# MAGIC - リアルタイム用途：REST API経由での低レイテンシー推論
# MAGIC
# MAGIC Databricksは両方のオプションを簡単に提供します。
# MAGIC
# MAGIC <!-- 利用データ収集（ビュー）。無効化するには削除してください。詳細はREADMEを参照。 -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F03-running-cv-inferences&demo_name=computer-vision-pcb&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fcomputer-vision-pcb%2F03-running-cv-inferences&version=1&user_hash=35036e02ef9687be687f67a76651cd8f60e5f14ae8134cd65342d7a5de24f158">

# COMMAND ----------

# MAGIC %md
# MAGIC ### このデモ用のクラスタが作成されています
# MAGIC このデモを実行するには、ドロップダウンメニューからクラスタ `dbdemos-computer-vision-pcb-yusuke_tsuchiya` を選択してください（[クラスタ設定を開く](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0727-093818-4rt9g6xe/configuration)）。<br />
# MAGIC *注: クラスタが30日後に削除された場合は、`dbdemos.create_cluster('computer-vision-pcb')` で再作成するか、`dbdemos.install('computer-vision-pcb')` でデモを再インストールできます。*

# COMMAND ----------

# MAGIC %md
# MAGIC ## バッチ & ストリーミング モデル推論
# MAGIC
# MAGIC まずはバッチ／ストリーミング推論から始めましょう。Sparkの分散処理機能を使って、スケールさせた推論を実行します。
# MAGIC
# MAGIC そのために、MLflowレジストリからモデルをロードし、Pandas UDFを作成して複数インスタンス（通常はGPU）で推論を分散実行します。
# MAGIC
# MAGIC 最初のステップは、モデルの依存関係をインストールし、同じライブラリバージョンでモデルをロードできるようにすることです。

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.39.0 mlflow==2.20.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Init the demo
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Load the pip requirements from the model registry
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os

# Use the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = f"{catalog}.{db}.dbdemos_pcb_classification"
MODEL_URI = f"models:/{MODEL_NAME}@Production"

# download model requirement from remote registry
requirements_path = ModelsArtifactRepository(MODEL_URI).download_artifacts(artifact_path="requirements.txt") 

if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

# COMMAND ----------

# DBTITLE 1,Install the requirements
# MAGIC %pip install -r $requirements_path
# MAGIC # 必要なライブラリをrequirements.txtからインストール

# COMMAND ----------

# DBTITLE 1,Load the model from the Registry
import torch
# GPUが利用可能な場合はGPUを活用
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

pipeline = mlflow.transformers.load_model(MODEL_URI, device=device.index)

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデル推論をローカル（非分散）で実行
# MAGIC
# MAGIC まずは標準のpandasデータフレームでローカル推論を実行してみましょう：

# COMMAND ----------

import io
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

# パイプラインを呼び出し、主要なクラスとその確率を返します
def predict_byte_series(content_as_byte_series, pipeline):
  # huggingfaceパイプライン用にPIL画像のリストへ変換
  image_list = content_as_byte_series.apply(lambda b: Image.open(io.BytesIO(b))).to_list()
  # パイプラインは全クラスの確率を返します
  predictions = pipeline.predict(image_list)
  # 最も高いスコアのクラスのみを抽出して返します [{'score': 0.999038815498352, 'label': 'normal'}, ...]
  return pd.DataFrame([max(r, key=lambda x: x['score']) for r in predictions])  

# Sparkテーブルからデータを取得
df = spark.read.table("training_dataset_augmented").limit(50)
# モデルを推論モードに切り替え
pipeline.model.eval()
with torch.set_grad_enabled(False):
  predictions = predict_byte_series(df.limit(10).toPandas()['content'], pipeline)
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SparkとPandas UDFで推論を分散実行（バッチ／ストリーミング推論）
# MAGIC
# MAGIC 関数をpandas UDFでラップし、推論を並列化してみましょう：

# COMMAND ----------

import numpy as np
import torch
from typing import Iterator

# 画像はメモリを消費するため、UDFで推論を1000件ずつバッチ処理
try:
  spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 1000)
except:
  pass

@pandas_udf("struct<score: float, label: string>")
def detect_damaged_pcb(images_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
  # パイプラインを推論モードに切り替え
  pipeline.model.eval()
  with torch.set_grad_enabled(False):
    for images in images_iter:
      yield predict_byte_series(images, pipeline)

# COMMAND ----------

display(df.limit(3).select('filename', 'content').withColumn("prediction", detect_damaged_pcb("content")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## REST APIによるリアルタイム推論
# MAGIC
# MAGIC 多くのユースケースではリアルタイム性が求められます。たとえば、PCB製造システムでのリアルタイム解析です。写真が撮影され、即座に欠陥の有無を判定する必要があります。
# MAGIC
# MAGIC これを実現するには、推論をREST API経由で提供する必要があります。システムは画像を送信し、エンドポイントが予測結果を返します。
# MAGIC
# MAGIC そのためには、画像バイト列をbase64でエンコードしてREST APIに送信します。
# MAGIC
# MAGIC この場合、モデル側でbase64をデコードしてPIL画像に戻す必要があります。これを簡単にするため、transformersパイプラインとbase64変換処理を持つカスタムモデルラッパーを作成します。
# MAGIC
# MAGIC 通常通り `mlflow.pyfunc.PythonModel` クラスを継承して実装します：

# COMMAND ----------

# DBTITLE 1,Model Wrapper for base64 image decoding
from io import BytesIO
import base64

# モデルラッパー
class RealtimeCVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        # モデルを推論モードで初期化
        self.pipeline.model.eval()

    # imagesにはbase64でエンコードされた画像が含まれます
    def predict(self, context, images):
        with torch.set_grad_enabled(False):
          # base64をPIL画像に変換
          images = images['data'].apply(lambda b: Image.open(BytesIO(base64.b64decode(b)))).to_list()
          # 予測を実行
          predictions = self.pipeline(images)
          # 最も高いスコアの予測を返す
          return pd.DataFrame([max(r, key=lambda x: x['score']) for r in predictions])

# COMMAND ----------

# DBTITLE 1,Load our image-based model, wrap it as base64-based, and test it
def to_base64(b):
  return base64.b64encode(b).decode("ascii")

# レジストリからモデルをロードし、最終的なhuggingfaceパイプラインを構築
pipeline_cpu = mlflow.transformers.load_model(MODEL_URI, return_type="pipeline", device=torch.device("cpu"))

# モデルをPyFuncModelとしてラップし、リアルタイムサービングエンドポイントで利用可能に
rt_model = RealtimeCVModelWrapper(pipeline_cpu)

# エンドポイントにデプロイする前に、ローカルで動作確認
# 入力をbase64を含むpandasデータフレームに変換（サーバーレスモデルエンドポイントが受け取る形式）
pdf = df.toPandas()
df_input = pd.DataFrame(pdf["content"].apply(to_base64).to_list(), columns=["data"])

predictions = rt_model.predict(None, df_input)
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ラッパーの準備ができたので、新しいモデルとしてレジストリにデプロイしましょう。
# MAGIC
# MAGIC リアルタイム推論が主な用途の場合、通常はトレーニングステップでモデルを登録する際にこの処理を行います。このデモでは、バッチ用とbase64ラッパー付きリアルタイム推論用の2つのモデルを作成します。

# COMMAND ----------

# DBTITLE 1,Save or RT model taking base64 in the registry
from mlflow.models.signature import infer_signature
DBDemos.init_experiment_for_batch("computer-vision-dl", "pcb")

with mlflow.start_run(run_name="hugging_face_rt") as run:
  signature = infer_signature(df_input, predictions)
  # モデルをMLflowにログ
  reqs = mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=rt_model, 
    pip_requirements=requirements_path, 
    input_example=df_input, 
    signature = signature)
  mlflow.set_tag("dbdemos", "pcb_classification")
  mlflow.set_tag("rt", "true")

# モデルをレジストリに登録
model_registered = mlflow.register_model(
  model_uri="runs:/"+run.info.run_id+"/model", 
  name="dbdemos_pcb_classification")

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルエンドポイントのデプロイ
# MAGIC
# MAGIC 新しいモデルラッパーがレジストリに登録されました。この新しいバージョンをモデルエンドポイントとしてデプロイし、リアルタイム推論を開始できます。

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

# キャプチャ用テーブル名のprefixをユニークにする（例: タイムスタンプやUUIDを付与）
# 以下エラーの回避
# BadRequest: Table in Unity Catalog ytsuchiya_demo.dbdemos_computer_vision_pcb.tsuchiya_dbdemos_pcb_classification_endpoint_payload already exists. Please specify a different table prefix.

import uuid
unique_prefix = f"{serving_endpoint_name}_payload_{uuid.uuid4().hex[:8]}"

# モデルサービングエンドポイントの設定を指定
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version=model_registered.version,
            workload_size="Small",
            workload_type="CPU",
            scale_to_zero_enabled=True
        )
    ],
    auto_capture_config = AutoCaptureConfigInput(
      catalog_name=catalog, 
      schema_name=db, 
      enabled=True,
      table_name_prefix=unique_prefix
    )
)

# 新しいバージョンをリリースする場合はTrueに設定（デモではデフォルトでエンドポイントを新しいモデルバージョンに更新しません）
force_update = False 

# 既存のエンドポイントがあるか確認
w = WorkspaceClient()
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
if existing_endpoint == None:
    print(f"エンドポイント {serving_endpoint_name} を作成します。パッケージングとデプロイに数分かかります...")
    from datetime import timedelta
    w.serving_endpoints.create_and_wait(
      name=serving_endpoint_name, 
      config=endpoint_config, 
      timeout=timedelta(minutes=60))
else:
  print(f"エンドポイント {serving_endpoint_name} は既に存在します...")
  if force_update:
    print(f"{endpoint_config.served_entities[0].entity_name} のバージョンを {endpoint_config.served_entities[0].entity_version} に更新します（エンドポイント: {serving_endpoint_name}）...")
    from datetime import timedelta
    w.serving_endpoints.update_config_and_wait(
      served_entities=endpoint_config.served_entities, 
      name=serving_endpoint_name,
      timeout=timedelta(minutes=60))

# COMMAND ----------

import base64
import pandas as pd

def to_base64(b):
  return base64.b64encode(b).decode("ascii")

# Sparkテーブルからデータを取得
df = spark.read.table(f"{catalog}.{db}.training_dataset_augmented").limit(50)

# エンドポイントにデプロイする前に、ローカルで動作確認
# 入力をbase64を含むpandasデータフレームに変換（サーバーレスモデルエンドポイントが受け取る形式）
pdf = df.toPandas()
df_input = pd.DataFrame(pdf["content"].apply(to_base64).to_list(), columns=["data"])
display(df_input)

# COMMAND ----------

import timeit
import mlflow
from mlflow import deployments

client = mlflow.deployments.get_deploy_client("databricks")

for i in range(3):
    input_slice = df_input[2*i:2*i+2]
    starting_time = timeit.default_timer()
    inferences = client.predict(
        endpoint=serving_endpoint_name, 
        inputs={
            "dataframe_records": input_slice.to_dict(orient='records')
        })
    print(f"推論時間（エンドツーエンド） :{round((timeit.default_timer() - starting_time), 3)}s")
    print("  "+str(inferences))

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC ## まとめ
# MAGIC
# MAGIC Databricksを使って、ディープラーニングモデルをスケールしてデプロイする方法、そしてDatabricks Serverless Model ServingでRESTエンドポイントとして提供する方法を紹介しました。
# MAGIC
# MAGIC ### 次のステップ：モデルの説明性
# MAGIC
# MAGIC 次は、モデルがどのピクセルを損傷と判断したかを説明・可視化する方法を学びましょう。
# MAGIC
# MAGIC [04-explaining-inferenceノートブック]($./04-explaining-inference)を開き、SHAPを使った予測の分析方法を確認してください。
# MAGIC
# MAGIC ### さらに進むには
# MAGIC
# MAGIC huggingfaceだけでは物足りない場合、pytorchなどのライブラリを活用したカスタム統合も可能です。
# MAGIC
# MAGIC [05-torch-lightning-training-and-inferenceノートブック]($./05-torch-lightning-training-and-inference)を開き、[PyTorch Lightning](https://www.pytorchlightning.ai/index.html)でPyTorchモデルを学習・デプロイする方法を学びましょう。

# COMMAND ----------

