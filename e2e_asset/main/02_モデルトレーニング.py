# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Hugging Faceでコンピュータビジョンモデルを構築する
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-4.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC 次のステップとして、データサイエンティストは画像セグメンテーションを実行するMLモデルを実装します。
# MAGIC
# MAGIC 前のデータパイプラインで作成したゴールドテーブルをトレーニングデータセットとして再利用します。
# MAGIC
# MAGIC このようなモデルの構築は、<a href="https://huggingface.co/docs/transformers/index">huggingface transformerライブラリ</a>を使うことで大幅に簡素化されます。
# MAGIC  
# MAGIC
# MAGIC ## MLOpsのステップ
# MAGIC
# MAGIC 画像セグメンテーションモデルの構築自体は簡単ですが、実運用環境にデプロイするのはより困難です。
# MAGIC
# MAGIC DatabricksはMLFlowの活用により、データサイエンスのプロセスを簡素化・加速します。
# MAGIC
# MAGIC * 自動実験管理とトラッキング
# MAGIC * hyperoptによる分散ハイパーパラメータチューニング
# MAGIC * MLFlowによるモデルパッケージ化（MLフレームワークを抽象化）
# MAGIC * ガバナンスのためのモデルレジストリ
# MAGIC * バッチまたはリアルタイム推論（ワンクリックデプロイ）
# MAGIC
# MAGIC <!-- 利用状況データの収集（閲覧）。無効化するには削除してください。詳細はREADMEを参照。 -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F02-huggingface-model-training&demo_name=computer-vision-pcb&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fcomputer-vision-pcb%2F02-huggingface-model-training&version=1&user_hash=35036e02ef9687be687f67a76651cd8f60e5f14ae8134cd65342d7a5de24f158">

# COMMAND ----------

# MAGIC %md
# MAGIC ### このデモ用のクラスタが作成されました
# MAGIC このデモを実行するには、ドロップダウンメニューから `dbdemos-computer-vision-pcb-yusuke_tsuchiya` クラスタを選択してください（[クラスタ設定を開く](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0727-093818-4rt9g6xe/configuration)）。<br />
# MAGIC *注: クラスタが30日後に削除された場合は、`dbdemos.create_cluster('computer-vision-pcb')` で再作成するか、`dbdemos.install('computer-vision-pcb')` でデモを再インストールできます。*

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.39.0 datasets==2.20.0 transformers==4.49.0 tf-keras==2.17.0 accelerate==1.4.0 mlflow==2.20.2 torchvision==0.20.1 deepspeed==0.14.4 evaluate==0.4.3
# MAGIC dbutils.library.restartPython()
# MAGIC
# MAGIC # トレーニング実験のセットアップ
# MAGIC DBDemos.init_experiment_for_batch("computer-vision-dl", "pcb")
# MAGIC
# MAGIC # ゴールドテーブルからデータを読み込み、表示
# MAGIC # ... 既存のコード ...

# COMMAND ----------

# DBTITLE 1,Demo Initialization
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Review our training dataset
#Setup the training experiment
DBDemos.init_experiment_for_batch("computer-vision-dl", "pcb")

df = spark.read.table("training_dataset_augmented")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deltaテーブルからデータセットを作成する
# MAGIC
# MAGIC Hugging Faceではこのステップも非常に簡単です。`Dataset.from_spark` 関数を呼び出すだけでOKです。
# MAGIC
# MAGIC 新しいDelta Loaderの詳細については、<a href="https://www.databricks.com/blog/contributing-spark-loader-for-hugging-face-datasets">ブログ記事</a>をご覧ください。

# COMMAND ----------

# DBTITLE 1,Create the transformer dataset from a spark dataframe (Delta Table)   
from datasets import Dataset

# 注意: from_sparkのサポートはserverless computeで提供予定です。このデモでは小規模データセットのためfrom_pandasを使用します
#dataset = Dataset.from_spark(df), cache_dir="/tmp/hf_cache/train").rename_column("content", "image")
dataset = Dataset.from_pandas(df.toPandas()).rename_column("content", "image")

# データセットをトレーニング用と検証用に分割
splits = dataset.train_test_split(test_size=0.2, seed = 42)
train_ds = splits['train']
val_ds = splits['test']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hugging Faceによる転移学習
# MAGIC
# MAGIC 転移学習とは、他のタスクで数千枚の画像に対して事前に学習された既存のモデルを利用し、その知識を自分たちのドメインに転用する手法です。Hugging Faceは、転移学習を非常に簡単に実装できるヘルパークラスを提供しています。
# MAGIC
# MAGIC 一般的なプロセスとしては、モデル全体または一部（通常は最後の層）を自分たちのカスタムデータセットで再学習（再トレーニング）します。
# MAGIC
# MAGIC これにより、トレーニングコストと効率の最適なバランスが得られます。特に、トレーニングデータセットが限られている場合に有効です。

# COMMAND ----------

# DBTITLE 1,Define the base model
import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor

# ファインチューニング元の事前学習済みモデル
# 詳細や他のモデルはhuggingfaceのリポジトリを参照: https://huggingface.co/google/vit-base-patch16-224
model_checkpoint = "google/vit-base-patch16-224"

# GPUの利用可否を確認
if not torch.cuda.is_available(): # GPUが利用不可の場合
  # CPUデモ用に小型モデルを利用
  model_checkpoint = "WinKawaks/vit-tiny-patch16-224" 
  print("モデル学習にはGPUクラスタを推奨します。CPUインスタンスは非常に遅くなります。Serverlessの場合はEnvironementタブでGPUを選択してください（プレビューが必要な場合あり）。")

# COMMAND ----------

# DBTITLE 1,Define image transformations for training & validation
from PIL import Image
import io
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, Resize, ToTensor, Lambda

# モデルの特徴抽出（前処理ステップの情報を含む。リサイズや正規化など）
# モデルパラメータを使うことで、入力サイズが異なる他モデルへの切り替えも容易です。
model_def = AutoFeatureExtractor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=model_def.image_mean, std=model_def.image_std)
byte_to_pil = Lambda(lambda b: Image.open(io.BytesIO(b)).convert("RGB"))

# トレーニングデータセットへの変換。クロップ処理を追加
train_transforms = Compose([byte_to_pil,
                            RandomResizedCrop((model_def.size['height'], model_def.size['width'])),
                            ToTensor(), # PIL画像をテンソルに変換
                            normalize
                           ])
# 検証用変換。画像を所定サイズにリサイズのみ
val_transforms = Compose([byte_to_pil,
                          Resize((model_def.size['height'], model_def.size['width'])),
                          ToTensor(),  # PIL画像をテンソルに変換
                          normalize
                         ])

# トレーニングデータセットにランダムリサイズ・変換を追加
def preprocess_train(batch):
    """train_transformsをバッチ全体に適用"""
    batch["image"] = [train_transforms(image) for image in batch["image"]]
    return batch

# 検証データセット
def preprocess_val(batch):
    """val_transformsをバッチ全体に適用"""
    batch["image"] = [val_transforms(image) for image in batch["image"]]
    return batch
  
# トレーニング/検証用の変換をセット
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# COMMAND ----------

# DBTITLE 1,Build our model from the pretrained model
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

# クラスラベルと値のマッピング（推論時に正しいラベルを出力するためにhuggingfaceが使用）
label2id, id2label = dict(), dict()
for i, label in enumerate(set(dataset['label'])):
    label2id[label] = i
    id2label[i] = label
    
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True # すでにファインチューニング済みのチェックポイントを再利用する場合は指定
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのファインチューニング
# MAGIC
# MAGIC データセットとモデルの準備ができました。ここからトレーニングステップを開始し、モデルをファインチューニングします。
# MAGIC
# MAGIC *本番環境レベルのユースケースでは、通常[ハイパーパラメータ](https://huggingface.co/docs/transformers/hpo_train)のチューニングを行いますが、今回はシンプルに固定設定で実行します。*

# COMMAND ----------

# DBTITLE 1,Training parameters
model_name = model_checkpoint.split("/")[-1]
batch_size = 32  # トレーニングと評価時のバッチサイズ

args = TrainingArguments(
    f"/tmp/huggingface/pcb/{model_name}-finetuned-leaf",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    no_cuda=not torch.cuda.is_available(),  # resnet用にCPUで実行（簡易化のため）
    num_train_epochs=20, 
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False
)

# COMMAND ----------

# DBTITLE 1,Define our evaluation metric
import numpy as np
import evaluate
# compute_metrics関数はNamed Tupleを入力として受け取ります:
# predictions: モデルのロジット（Numpy配列）
# label_ids: 正解ラベル（Numpy配列）

# F1スコアでモデルを評価します。このデモではバイナリ分類（デフォルトタイプで分類しません）
accuracy = evaluate.load("f1")

def compute_metrics(eval_pred):
    """予測バッチに対する精度を計算"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

# COMMAND ----------

# DBTITLE 1,Start our Training and log the model to MLFlow
import mlflow
from mlflow.models.signature import infer_signature
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from transformers import pipeline, DefaultDataCollator, EarlyStoppingCallback

def collate_fn(examples):
    pixel_values = torch.stack([e["image"] for e in examples])
    labels = torch.tensor([label2id[e["label"]] for e in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# モデルをGPUで学習するように設定
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

with mlflow.start_run(run_name="hugging_face") as run:
  early_stop = EarlyStoppingCallback(early_stopping_patience=10)
  trainer = Trainer(
    model, 
    args, 
    train_dataset=train_ds, 
    eval_dataset=val_ds, 
    tokenizer=model_def, 
    compute_metrics=compute_metrics, 
    data_collator=collate_fn, 
    callbacks = [early_stop])

  train_results = trainer.train()

  # 最終的なhugging faceパイプラインを構築
  classifier = pipeline(
    "image-classification", 
    model=trainer.state.best_model_checkpoint, 
    tokenizer = model_def, 
    device_map='auto')
  
  # MLFlowにモデルを記録
  #    pip_requirementsは任意。依存関係をカスタム指定する場合に利用
  reqs = mlflow.transformers.get_default_pip_requirements(model)

  #    signatureは入出力スキーマを指定。単一予測で出力スキーマを取得
  transform = ToPILImage()
  img = transform(val_ds[0]['image'])
  prediction = classifier(img)
  signature = infer_signature(
    model_input=np.array(img), 
    model_output=pd.DataFrame(prediction))
  
  #    モデルを記録し、タグ・メトリクスも記録
  mlflow.transformers.log_model(
    artifact_path="model", 
    transformers_model=classifier, 
    pip_requirements=reqs,
    signature=signature)
  
  mlflow.set_tag("dbdemos", "pcb_classification")
  mlflow.log_metrics(train_results.metrics)

  #    入力データセットも記録し、テーブルからモデルへのリネージを追跡
  src_dataset = mlflow.data.load_delta(
    table_name=f'{catalog}.{db}.training_dataset_augmented')
  mlflow.log_input(src_dataset, context="Training-Input")

# COMMAND ----------

# DBTITLE 1,Let's try our model to make sure it works as expected
import json
import io
from PIL import Image

def test_image(test, index):
  img = Image.open(io.BytesIO(test.iloc[index]['content']))
  print("ファイル名: " + test.iloc[index]['filename'])
  print("正解ラベル: " + test.iloc[index]['label'])
  print(f"予測結果: {json.dumps(classifier(img), indent=4)}")
  display(img)

# トレーニングデータセットから'normal'ラベルと'damaged'ラベルの画像をサンプル
normal_samples = spark.read.table("training_dataset_augmented").filter("label == 'normal'").select("content", "filename", "label").limit(10).toPandas()
damaged_samples = spark.read.table("training_dataset_augmented").filter("label == 'damaged'").select("content", "filename", "label").limit(10).toPandas()

# 各グループの最初の画像でモデルをテスト
test_image(normal_samples, 0)
print('\n\n=======================')
test_image(damaged_samples, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのデプロイ
# MAGIC
# MAGIC モデルのトレーニングが完了しました。あとはModel Registryに保存し、「Production ready」として移行するだけです。<br/>
# MAGIC このデモでは最新のランを使用しますが、トレーニング時に定義した指標に基づいて `mlflow.search_runs` で最良のランを検索することもできます。

# COMMAND ----------

# DBTITLE 1,Save the model in the registry & mark it for Production
# Unity Catalogにモデルを登録
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = f"{catalog}.{db}.dbdemos_pcb_classification"

model_registered = mlflow.register_model("runs:/"+run.info.run_id+"/model", MODEL_NAME)
print("モデルバージョン"+model_registered.version+"をProductionモデルとして登録します")

## モデルバージョンにProductionエイリアスを付与
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
  name = MODEL_NAME, 
  version = model_registered.version,
  alias = "Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルレジストリ
# MAGIC Unity Catalogのモデルレジストリでモデルを確認しましょう。
# MAGIC
# MAGIC 1. 左側のナビゲーションメニューから「カタログエクスプローラー」を開きます。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-model-lineage-01.png?raw=true"/>
# MAGIC
# MAGIC 2. 検索ボックスやカタログブラウザを使って、カタログとスキーマ内の `dbdemos_pcb_classification` モデルを探します。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-model-lineage-02.png?raw=true"/>
# MAGIC
# MAGIC 3. エイリアス `@production` が付与されたバージョンを開きます。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-model-lineage-03.png?raw=true"/>
# MAGIC
# MAGIC 4. 「Lineage（リネージ）」タブを選択します。`training_dataset_augmented` テーブルがモデルの上流接続として表示されていることを確認できます。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-model-lineage-04.png?raw=true"/>
# MAGIC
# MAGIC 5. 「See lineage graph（リネージグラフを表示）」をクリックします。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-model-lineage-05.png?raw=true"/>
# MAGIC
# MAGIC 6. 拡張アイコンやカラム名を使って、ボリュームから取り込まれた生データまでモデルのリネージをたどってみましょう。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-model-lineage.png?raw=true"/>
# MAGIC
# MAGIC Unity Catalogのリネージ機能は、データの流れや利用状況を生データの取り込みからモデル学習までエンドツーエンドで可視化します。リネージ情報は[システムテーブル](https://docs.databricks.com/ja/admin/system-tables/lineage.html)やUIから確認できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 次へ: バッチおよびリアルタイム推論
# MAGIC
# MAGIC モデルはトレーニングされ、MLflow Model Registryに登録されました。Databricksを使うことで、モデルのトレーニングに必要な多くの補助的なコードを削減でき、モデル性能の向上に集中できます。
# MAGIC
# MAGIC 次のステップは、このモデルを使ってバッチまたはRESTエンドポイント経由のリアルタイム推論を行うことです。
# MAGIC
# MAGIC Databricksのサービング機能を活用した推論方法は、次の[03-running-cv-inferencesノートブック]($./03-running-cv-inferences)でご確認ください。

# COMMAND ----------

