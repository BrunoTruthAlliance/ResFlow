from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName(
    f"Get final csvs").enableHiveSupport().getOrCreate()
sc = spark.sparkContext

df = spark.read.parquet(sys.argv[1])
column_names = df.columns
bucket_columns = [col for col in column_names if col.startswith("id_")]
bucket_columns.remove("id_play_time_ms")
bucket_columns.remove("id_duration_ms")
index_columns = [col for col in column_names if col.startswith(
    "index_") or col.startswith("onehot_feat")]
other_columns = [
    "is_video_author",
    "music_type",
    "date"
]
feature_columns = bucket_columns + index_columns + other_columns
label_columns = ["is_click", "is_like", "is_follow",
                 "is_comment", "is_forward", "play_time_ms"]
df_final = df\
    .select(
        [F.col(c).cast('int').alias(c) if c.startswith("id_") else F.col(c) for c in feature_columns +
         label_columns] + [(F.col("play_time_ms") / F.lit(1000.)).cast('int').alias("play_time_s")]
    )\
    .filter(F.col("play_time_ms") > 0)\
    .filter(F.col("tab").isNotNull() & (F.col("tab") == 1))
df_final = df_final.cache()
# df_final.summary().show(truncate=False)
df_final.select(F.percentile_approx(
    "play_time_s", [(1.0+i)/10 for i in range(0, 9)])).show(truncate=False)
df_final.select(
    F.mean("is_click"),
    F.mean("is_like"),
    F.mean("is_follow"),
    F.mean("is_comment"),
    F.mean("is_forward")
).show(truncate=False)
df_final.write.partitionBy("date").mode(
    "overwrite").csv(sys.argv[2], header=True)
