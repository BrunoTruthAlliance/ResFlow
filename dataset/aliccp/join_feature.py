from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName(
    'join csv feature to table').enableHiveSupport().getOrCreate()

train_sample = spark.read.csv(sys.argv[1], header=True, inferSchema=True)
train_user_feature = spark.read.csv(sys.argv[2], header=True, inferSchema=True)

test_sample = spark.read.csv(sys.argv[3], header=True, inferSchema=True)
test_user_feature = spark.read.csv(sys.argv[4], header=True, inferSchema=True)

train_full = train_sample.join(
    train_user_feature, on="common_feature_index", how="inner")
train_full.write.mode("overwrite").saveAsTable(sys.argv[5])

test_full = test_sample.join(
    test_user_feature, on="common_feature_index", how="inner")
test_full.write.mode("overwrite").saveAsTable(sys.argv[6])
