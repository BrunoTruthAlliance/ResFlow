from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys

spark = SparkSession.builder.appName(
    "Read kuairand pure user features and save as parquet").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
# Define the PySpark schema
schema = StructType([
    StructField("user_id", LongType()),
    StructField("user_active_degree", StringType()),
    StructField("is_lowactive_period", IntegerType()),
    StructField("is_live_streamer", IntegerType()),
    StructField("is_video_author", IntegerType()),
    StructField("follow_user_num", IntegerType()),
    StructField("follow_user_num_range", StringType()),
    StructField("fans_user_num", IntegerType()),
    StructField("fans_user_num_range", StringType()),
    StructField("friend_user_num", IntegerType()),
    StructField("friend_user_num_range", StringType()),
    StructField("register_days", IntegerType()),
    StructField("register_days_range", StringType()),
    StructField("onehot_feat0", IntegerType()),
    StructField("onehot_feat1", IntegerType()),
    StructField("onehot_feat2", IntegerType()),
    StructField("onehot_feat3", IntegerType()),
    StructField("onehot_feat4", DoubleType()),
    StructField("onehot_feat5", IntegerType()),
    StructField("onehot_feat6", IntegerType()),
    StructField("onehot_feat7", IntegerType()),
    StructField("onehot_feat8", IntegerType()),
    StructField("onehot_feat9", IntegerType()),
    StructField("onehot_feat10", IntegerType()),
    StructField("onehot_feat11", IntegerType()),
    StructField("onehot_feat12", DoubleType()),
    StructField("onehot_feat13", DoubleType()),
    StructField("onehot_feat14", DoubleType()),
    StructField("onehot_feat15", DoubleType()),
    StructField("onehot_feat16", DoubleType()),
    StructField("onehot_feat17", DoubleType())
])


data_path = sys.argv[1]
df = spark.read.csv(data_path, header=True, schema=schema)
df.repartition(1).write.mode("overwrite").parquet(sys.argv[2])
