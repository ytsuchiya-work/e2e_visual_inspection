# Databricks notebook source
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,画像の読み込みと確認
import os
from PIL import Image
import matplotlib.pyplot as plt

NUM_SUFFIX = 1

def read_img(path):
    return Image.open(path)

def display_image(img, comment, dpi=300):
    width, height = img.size
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.title(comment)
    plt.imshow(img, interpolation="nearest", aspect="auto")

cwd = os.getcwd()
normal_img = read_img(f"{cwd}/image/normal/normal_00{NUM_SUFFIX}.JPG")
display_image(normal_img, "Normal image")
anomaly_img = read_img(f"{cwd}/image/anomaly/anomaly_00{NUM_SUFFIX}.JPG")
display_image(anomaly_img, "Anomaly image")

# COMMAND ----------

# DBTITLE 1,推論用の前処理
IMAGE_RESIZE = 256

def resize_image(img):
    width, height = img.size   # Get dimensions
    new_size = min(width, height)
    # Crop the center of the img
    img = img.crop(((width - new_size)/2, (height - new_size)/2, (width + new_size)/2, (height + new_size)/2))
    #Resize to the new resolution
    img = img.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
    return img

normal_img_resize = resize_image(normal_img)
display_image(normal_img_resize, "Resized normal image")
anomaly_img_resize = resize_image(anomaly_img)
display_image(anomaly_img_resize, "Resized anomaly image")

# COMMAND ----------

import io
import timeit
import base64
import mlflow
from mlflow import deployments

client = mlflow.deployments.get_deploy_client("databricks")

def convert_PIL_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

# 入力をbase64に変換（サーバーレスモデルエンドポイントが受け取る形式）
def to_base64(byte):
    return base64.b64encode(byte).decode("ascii")

def predict(img):
    t_start = timeit.default_timer()
    img_bytes = convert_PIL_to_bytes(img)
    inferences = client.predict(
        endpoint = serving_endpoint_name, 
        inputs = {
            "inputs": {"data": to_base64(img_bytes)}
        })
    print(f"推論時間（エンドツーエンド） :{round((timeit.default_timer() - t_start)*1000)}ms")
    print("  "+str(inferences))

predict(normal_img_resize)
predict(anomaly_img_resize)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC