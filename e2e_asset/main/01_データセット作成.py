# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # PCB - インジェストデータパイプライン
# MAGIC
# MAGIC このパイプラインでは、次の2つのデータセットを取り込みます。
# MAGIC
# MAGIC * PCBを含む生画像（jpg）
# MAGIC * CSVファイルとして保存された異常タイプのラベル
# MAGIC
# MAGIC まず、このデータを段階的にロードし、最終的なGoldテーブルを作成するデータパイプラインの構築に注力します。
# MAGIC
# MAGIC このテーブルは、リアルタイムで画像の異常を検出するML分類モデルのトレーニングに使用されます！
# MAGIC
# MAGIC *このデモは標準のSpark APIを活用しています。同じパイプラインは、Delta Live Tablesを使って純粋なSQLでも実装可能です。DLTの詳細については、`dbdemos.install('dlt-loans')`をインストールしてください。*
# MAGIC
# MAGIC <!-- 利用状況データ（ビュー）の収集。無効にするには削除してください。詳細はREADMEを参照。 -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-ingestion-and-ETL&demo_name=computer-vision-pcb&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fcomputer-vision-pcb%2F01-ingestion-and-ETL&version=1&user_hash=35036e02ef9687be687f67a76651cd8f60e5f14ae8134cd65342d7a5de24f158">

# COMMAND ----------

# MAGIC %md
# MAGIC ### このデモ用のクラスタが作成されました
# MAGIC このデモを実行するには、ドロップダウンメニューからクラスタ `dbdemos-computer-vision-pcb-yusuke_tsuchiya` を選択してください（[クラスタ設定を開く](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0727-093818-4rt9g6xe/configuration)）。<br />
# MAGIC *注: クラスタが30日後に削除された場合は、`dbdemos.create_cluster('computer-vision-pcb')` で再作成するか、`dbdemos.install('computer-vision-pcb')` でデモを再インストールできます。*

# COMMAND ----------

# pip install dbdemos

# COMMAND ----------

# import dbdemos
# dbdemos.create_cluster('computer-vision-pcb')

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.39.0 mlflow==2.20.2 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

print(f"Training data has been installed in the volume {volume_folder}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## インカミングデータセットの確認
# MAGIC
# MAGIC データセットは自動的にダウンロードされ、DBFSストレージフォルダに保存されています。データを確認してみましょう。

# COMMAND ----------

display(dbutils.fs.ls(f"{volume_folder}/images/Normal/"))
display(dbutils.fs.ls(f"{volume_folder}/labels/"))
display(dbutils.fs.head(f"{volume_folder}/labels/image_anno.csv"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCB画像の確認
# MAGIC
# MAGIC 画像は`matplotlib`を使ってPythonで表示できます。
# MAGIC
# MAGIC まず正常な画像を確認し、その後異常のある画像を見てみましょう。

# COMMAND ----------

from PIL import Image
import matplotlib.pyplot as plt

def display_image(path, dpi=300):
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.imshow(img, interpolation="nearest", aspect="auto")


display_image(f"{volume_folder}/images/Normal/0000.JPG")
display_image(f"{volume_folder}/images/Anomaly/000.JPG")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Databricks Autoloaderで生画像をインジェスト
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-1.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC 最初のステップは、個々のJPG画像をロードすることです。特に増分ロード（新しい画像のみを取り込む）では、大規模な場合は困難になることがあります。
# MAGIC
# MAGIC Databricks Autoloaderは、あらゆる形式のデータを簡単に扱うことができ、新しいデータセットのインジェストも非常に簡単です。
# MAGIC
# MAGIC Autoloaderは、新しいファイルのみを確実に処理し、数百万枚の画像にもスケールします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto Loaderでバイナリファイルを読み込む
# MAGIC
# MAGIC Auto Loaderを使って画像を読み込み、Spark関数でラベル列を作成できます。Auto Loaderはテーブルを自動的に作成し、バイナリ用に圧縮を無効化するなど最適化も行います。
# MAGIC
# MAGIC また、画像とラベルの内容をテーブルとして簡単に表示することもできます。

# COMMAND ----------

(spark.readStream.format("cloudFiles")
                 .option("cloudFiles.format", "binaryFile")
                 .option("pathGlobFilter", "*.JPG")
                 .option("recursiveFileLookup", "true")
                 .option("cloudFiles.schemaLocation", f"{volume_folder}/stream/pcb_schema")
                 .option("cloudFiles.maxFilesPerTrigger", 200)
                 .load(f"{volume_folder}/images/")
    .withColumn("filename", F.substring_index(col("path"), "/", -1))
    .writeStream.trigger(availableNow=True)
                .option("checkpointLocation", f"{volume_folder}/stream/pcb_checkpoint")
                .toTable("pcb_images").awaitTermination())

spark.sql("ALTER TABLE pcb_images OWNER TO `account users`")
display(spark.table("pcb_images"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Auto LoaderでCSVラベルファイルを読み込む
# MAGIC CSVファイルはDatabricksの[Auto Loader](https://docs.databricks.com/ja/ingestion/auto-loader/index.html)を使って簡単に読み込むことができ、スキーマの推論や進化にも対応しています。

# COMMAND ----------

(spark.readStream.format("cloudFiles")
                 .option("cloudFiles.format", "csv")
                 .option("header", True)
                 .option("cloudFiles.schemaLocation", f"{volume_folder}/stream/labels_schema")
                 .load(f"{volume_folder}/labels/")
      .withColumn("filename", F.substring_index(col("image"), "/", -1))
      .select("filename", "label")
      .withColumnRenamed("label", "labelDetail")
      .writeStream.trigger(availableNow=True)
                  .option("checkpointLocation", f"{volume_folder}/stream/labels_checkpoint")
                  .toTable("pcb_labels").awaitTermination())

spark.sql("ALTER TABLE pcb_labels SET OWNER TO `account users`")
display(spark.table("pcb_labels"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ラベルテーブルと画像テーブルを結合しましょう
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-2.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC 取り込みを簡単にするためにDeltaテーブルを使っています。
# MAGIC
# MAGIC 個々の小さな画像を気にする必要はもうありません。
# MAGIC
# MAGIC この結合操作はPythonでもSQLでも実行できます。

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE training_dataset AS 
# MAGIC   (SELECT 
# MAGIC       *, 
# MAGIC       CASE WHEN labelDetail = 'normal' THEN 'normal' ELSE 'damaged' END as label
# MAGIC    FROM 
# MAGIC       pcb_images 
# MAGIC     INNER JOIN pcb_labels USING (filename)
# MAGIC   );
# MAGIC
# MAGIC ALTER TABLE training_dataset SET OWNER TO `account users`;
# MAGIC
# MAGIC SELECT * FROM training_dataset LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   label,
# MAGIC   count(*)
# MAGIC FROM
# MAGIC   training_dataset
# MAGIC GROUP BY
# MAGIC   label

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 最終ステップ: DLファインチューニング用の画像データセットの準備と拡張
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/computer-vision/deeplearning-cv-pcb-flow-3.png?raw=true" width="700px" style="float: right"/>
# MAGIC
# MAGIC 私たちはテーブルを使って作業しています。これらの変換はPythonやSQLで実行できます。
# MAGIC
# MAGIC 画像に対する一部の変換処理は高コストになる場合があります。Sparkを活用して画像の前処理を分散実行できます。
# MAGIC
# MAGIC この例では、以下の処理を行います:
# MAGIC - 画像の中心をクロップして正方形にします（ファインチューニングに使うモデルは正方形画像を想定しています）
# MAGIC - 画像を小さい解像度（256x256）にリサイズします。モデルは高解像度画像を使いません。
# MAGIC
# MAGIC また、データセットの拡張（オーグメンテーション）も行い、「異常」画像を増やします。ここではデータがかなり不均衡（10枚に1枚しか異常がない）です。<br/>
# MAGIC システムはPCBの写真を上下逆さまでも撮影しているようなので、推論時も同様の状況が想定されます。そこで、異常画像をすべて左右反転し、データセットに追加します。
# MAGIC
# MAGIC *注: deltatorchを使う場合は、ここで直接テスト/トレーニングデータセットの分割やidカラムの追加も可能です。詳細は 04-ADVANCED-pytorch-training-and-inference ノートブックを参照してください。*

# COMMAND ----------

# DBTITLE 1,Crop and resize our images
from PIL import Image
import io
from pyspark.sql.functions import pandas_udf
IMAGE_RESIZE = 256

#Resize UDF function
@pandas_udf("binary")
def resize_image_udf(content_series):
  def resize_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    width, height = image.size   # Get dimensions
    new_size = min(width, height)
    # Crop the center of the image
    image = image.crop(((width - new_size)/2, (height - new_size)/2, (width + new_size)/2, (height + new_size)/2))
    #Resize to the new resolution
    image = image.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(resize_image)


# add the metadata to enable the image preview
image_meta = {"spark.contentAnnotation" : '{"mimeType": "image/jpeg"}'}

(spark.table("training_dataset")
      .withColumn("sort", F.rand()).orderBy("sort").drop('sort') #shuffle the DF
      .withColumn("content", resize_image_udf(col("content")).alias("content", metadata=image_meta))
      .write.mode('overwrite').saveAsTable("training_dataset_augmented"))

spark.sql("ALTER TABLE training_dataset_augmented OWNER TO `account users`")

# COMMAND ----------

# DBTITLE 1,Flip and add damaged images
import PIL
@pandas_udf("binary")
def flip_image_horizontal_udf(content_series):
  def flip_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    #Flip
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(flip_image)

(spark.table("training_dataset_augmented")
    .filter("label == 'damaged'")
    .withColumn("content", flip_image_horizontal_udf(col("content")).alias("content", metadata=image_meta))
    .write.mode('append').saveAsTable("training_dataset_augmented"))

# COMMAND ----------

# DBTITLE 1,Final dataset now has 20% damaged images
# MAGIC %sql
# MAGIC SELECT
# MAGIC   label,
# MAGIC   count(*)
# MAGIC FROM
# MAGIC   training_dataset_augmented
# MAGIC GROUP BY
# MAGIC   label

# COMMAND ----------

# MAGIC %md
# MAGIC # Genie 用にデータを拡張

# COMMAND ----------

from pyspark.sql.functions import rand, when, lit

# テーブルを読み込み
df = spark.table(f"{catalog}.{db}.training_dataset")

# 選択肢
facilities = ["A", "B", "C", "D", "E"]
lines = ["Line1", "Line2", "Line3", "Line4", "Line5"]

# ランダムなインデックスを生成
df_new = df.withColumn("rand_idx", (rand() * 5).cast("int"))
df_new = df_new.withColumn("productionLine", 
        when(df_new["rand_idx"] == 0, lit(lines[0]))
        .when(df_new["rand_idx"] == 1, lit(lines[1]))
        .when(df_new["rand_idx"] == 2, lit(lines[2]))
        .when(df_new["rand_idx"] == 3, lit(lines[3]))
        .otherwise(lit(lines[4]))
    ) \
    .drop("rand_idx")

# 新しいテーブルとして保存
df_new.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{db}.training_dataset_append_new_columns")

# COMMAND ----------

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
SET TBLPROPERTIES (
  'comment' = 'PCBトレーニングデータセット。productionLine列を追加。'
)
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN filename COMMENT '画像ファイル名'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN path COMMENT 'ファイルパス'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN modificationTime COMMENT 'ファイルの最終更新日時'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN length COMMENT 'ファイルサイズ（バイト）'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN content COMMENT '画像バイナリデータ'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN labelDetail COMMENT '詳細なラベル情報'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN label COMMENT '分類ラベル'
""")

spark.sql("""
ALTER TABLE {catalog}.{db}.training_dataset_append_new_columns
ALTER COLUMN productionLine COMMENT '生産ライン名（Line1〜Line5）'
""")