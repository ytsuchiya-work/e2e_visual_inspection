# Databricks notebook source
# MAGIC %md
# MAGIC ## 手順
# MAGIC ### 1. Genie Space の作成
# MAGIC - 使用データ: training_dataset
# MAGIC - 指示: 
# MAGIC   - 日本語で回答と回答してください。
# MAGIC   - 分からない場合は、「分かりません」と回答してください。
# MAGIC - Description: PCB基盤の製造品質データについて確認するためのGenieスペースです。
# MAGIC - サンプル質問:
# MAGIC   - これまでで正常な製品と異常な製品はそれぞれ何個ずつ作成されましたか？
# MAGIC   - これまでの異常な製品の異常理由を全て教えて。
# MAGIC   - 各異常はそれぞれ何個生じていますか？
# MAGIC   
# MAGIC ### 2. Knowledge Assistant の作成
# MAGIC - 説明: 生産ラインのメンテナンスとPCB基盤の異常原因およびその解決策に関する質問に回答
# MAGIC - データセット:
# MAGIC   - pdf_data/pcb_anomaly_analysis.pdf
# MAGIC     - 説明: プリント基板の代表的な異常である溶融不良、傷、部品欠損、曲がり・変形について、発生原因や対策を詳細に解説しています。リフロー工程の温度管理や部品実装、搬送や保管工程など各段階での具体的なトラブル例と、それに対する予防・改善方法が網羅的に整理されています。設計・製造・検査体制の最適化の重要性を強調しています。
# MAGIC   - pdf_data/maintenace_report.pdf
# MAGIC     - 説明: 本資料はプリント基板生産ライン1〜5の定期メンテナンスレポートで、各ラインごとに搬送ベルトやはんだ付けヘッド温度、フラックス塗布、エアー圧などの確認項目、不具合兆候、実施済み対策をまとめています。
# MAGIC
# MAGIC ### 3. Multi Agent Superviser の作成
# MAGIC - 説明: 異常分析を行うためのスーパーバイザー
# MAGIC - エージェント:
# MAGIC   - Genie Space: Product_Quality_Check
# MAGIC   - Knowledge Assistant: ka-332418cb-endpoint

# COMMAND ----------

