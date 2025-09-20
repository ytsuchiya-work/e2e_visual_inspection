# Databricks notebook source
# MAGIC %md
# MAGIC # 03_main の実行に必要なクラスターの作成

# COMMAND ----------

pip install dbdemos

# COMMAND ----------

import dbdemos
dbdemos.create_cluster('computer-vision-pcb')

# COMMAND ----------

# MAGIC %md
# MAGIC 作成されたクラスターを使用して、02_main を実行

# COMMAND ----------

