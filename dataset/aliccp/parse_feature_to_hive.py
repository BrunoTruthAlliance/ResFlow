from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import sys


def parse_key_value(input_str):
    key_values = input_str.split(chr(0x01))
    result = []
    for key_value in key_values:
        kv = key_value.split(chr(0x02))
        if len(kv) == 2:
            result.append((kv[0], kv[1]))
    return result


parse_udf = F.udf(parse_key_value, ArrayType(StructType([
    StructField("feature_name", StringType(), True),
    StructField("feature_kv", StringType(), True)
])))


def get_keys(input_arr):
    result = [item.split(chr(0x03))[0] for item in input_arr]
    result_str = ";".join(result)
    return result_str


get_keys_udf = F.udf(get_keys, StringType())


def get_vals(input_arr):
    result = [item.split(chr(0x03))[1] for item in input_arr]
    result_str = ";".join(result)
    return result_str


get_vals_udf = F.udf(get_vals, StringType())

spark = SparkSession.builder.appName(
    'parse csv ori data to table').enableHiveSupport().getOrCreate()

train_sample = spark.read.csv(sys.argv[1], header=False, inferSchema=False)
train_sample = train_sample.withColumn("parsed", parse_udf(F.col("_c5"))) \
    .select(
        F.col("_c0").alias("sample_id"),
        F.col("_c1").alias("ctr_label"),
        F.col("_c2").alias("cvr_label"),
        F.col("_c3").alias("common_feature_index"),
        F.explode("parsed").alias("key_values")
) \
    .groupBy("sample_id", "ctr_label", "cvr_label", "common_feature_index") \
    .pivot("key_values.feature_name") \
    .agg(F.collect_list("key_values.feature_kv").alias("feature_kv")) \
    .withColumn("205_id", get_keys_udf(F.col("205"))) \
    .withColumn("206_id", get_keys_udf(F.col("206"))) \
    .withColumn("207_id", get_keys_udf(F.col("207"))) \
    .withColumn("210_id", get_keys_udf(F.col("210"))) \
    .withColumn("216_id", get_keys_udf(F.col("216"))) \
    .withColumn("301_id", get_keys_udf(F.col("301"))) \
    .withColumn("508_id", get_keys_udf(F.col("508"))) \
    .withColumn("508_val", get_vals_udf(F.col("508"))) \
    .withColumn("509_id", get_keys_udf(F.col("509"))) \
    .withColumn("509_val", get_vals_udf(F.col("509"))) \
    .withColumn("702_id", get_keys_udf(F.col("702"))) \
    .withColumn("702_val", get_vals_udf(F.col("702"))) \
    .withColumn("853_id", get_keys_udf(F.col("853"))) \
    .withColumn("853_val", get_vals_udf(F.col("853"))) \
    .select(
        F.col("sample_id"),
        F.col("ctr_label"),
        F.col("cvr_label"),
        F.col("common_feature_index"),
        F.col("205_id"),
        F.col("206_id"),
        F.col("207_id"),
        F.col("210_id"),
        F.col("216_id"),
        F.col("301_id"),
        F.col("508_id"),
        F.col("508_val"),
        F.col("509_id"),
        F.col("509_val"),
        F.col("702_id"),
        F.col("702_val"),
        F.col("853_id"),
        F.col("853_val")
) \

train_user_feature = spark.read.csv(
    sys.argv[2], header=False, inferSchema=False)
train_user_feature = train_user_feature.withColumn("parsed", parse_udf(F.col("_c2"))) \
    .select(
        F.col("_c0").alias("common_feature_index"),
        F.explode("parsed").alias("key_values")
) \
    .groupBy("common_feature_index") \
    .pivot("key_values.feature_name") \
    .agg(F.collect_list("key_values.feature_kv").alias("feature_kv")) \
    .withColumn("101_id", get_keys_udf(F.col("101"))) \
    .withColumn("121_id", get_keys_udf(F.col("121"))) \
    .withColumn("122_id", get_keys_udf(F.col("122"))) \
    .withColumn("124_id", get_keys_udf(F.col("124"))) \
    .withColumn("125_id", get_keys_udf(F.col("125"))) \
    .withColumn("126_id", get_keys_udf(F.col("126"))) \
    .withColumn("127_id", get_keys_udf(F.col("127"))) \
    .withColumn("128_id", get_keys_udf(F.col("128"))) \
    .withColumn("129_id", get_keys_udf(F.col("129"))) \
    .withColumn("109_14_id", get_keys_udf(F.col("109_14"))) \
    .withColumn("109_14_val", get_vals_udf(F.col("109_14"))) \
    .withColumn("110_14_id", get_keys_udf(F.col("110_14"))) \
    .withColumn("110_14_val", get_vals_udf(F.col("110_14"))) \
    .withColumn("127_14_id", get_keys_udf(F.col("127_14"))) \
    .withColumn("127_14_val", get_vals_udf(F.col("127_14"))) \
    .withColumn("150_14_id", get_keys_udf(F.col("150_14"))) \
    .withColumn("150_14_val", get_vals_udf(F.col("150_14"))) \
    .select(
        F.col("common_feature_index"),
        F.col("101_id"),
        F.col("121_id"),
        F.col("122_id"),
        F.col("124_id"),
        F.col("125_id"),
        F.col("126_id"),
        F.col("127_id"),
        F.col("128_id"),
        F.col("129_id"),
        F.col("109_14_id"),
        F.col("109_14_val"),
        F.col("110_14_id"),
        F.col("110_14_val"),
        F.col("127_14_id"),
        F.col("127_14_val"),
        F.col("150_14_id"),
        F.col("150_14_val")
)

train_sample.join(train_user_feature, on="common_feature_index", how="inner") \
    .write.mode("overwrite").saveAsTable(sys.argv[5])


test_sample = spark.read.csv(sys.argv[3], header=False, inferSchema=False)
test_sample = test_sample.withColumn("parsed", parse_udf(F.col("_c5"))) \
    .select(
        F.col("_c0").alias("sample_id"),
        F.col("_c1").alias("ctr_label"),
        F.col("_c2").alias("cvr_label"),
        F.col("_c3").alias("common_feature_index"),
        F.explode("parsed").alias("key_values")
) \
    .groupBy("sample_id", "ctr_label", "cvr_label", "common_feature_index") \
    .pivot("key_values.feature_name") \
    .agg(F.collect_list("key_values.feature_kv").alias("feature_kv")) \
    .withColumn("205_id", get_keys_udf(F.col("205"))) \
    .withColumn("206_id", get_keys_udf(F.col("206"))) \
    .withColumn("207_id", get_keys_udf(F.col("207"))) \
    .withColumn("210_id", get_keys_udf(F.col("210"))) \
    .withColumn("216_id", get_keys_udf(F.col("216"))) \
    .withColumn("301_id", get_keys_udf(F.col("301"))) \
    .withColumn("508_id", get_keys_udf(F.col("508"))) \
    .withColumn("508_val", get_vals_udf(F.col("508"))) \
    .withColumn("509_id", get_keys_udf(F.col("509"))) \
    .withColumn("509_val", get_vals_udf(F.col("509"))) \
    .withColumn("702_id", get_keys_udf(F.col("702"))) \
    .withColumn("702_val", get_vals_udf(F.col("702"))) \
    .withColumn("853_id", get_keys_udf(F.col("853"))) \
    .withColumn("853_val", get_vals_udf(F.col("853"))) \
    .select(
        F.col("sample_id"),
        F.col("ctr_label"),
        F.col("cvr_label"),
        F.col("common_feature_index"),
        F.col("205_id"),
        F.col("206_id"),
        F.col("207_id"),
        F.col("210_id"),
        F.col("216_id"),
        F.col("301_id"),
        F.col("508_id"),
        F.col("508_val"),
        F.col("509_id"),
        F.col("509_val"),
        F.col("702_id"),
        F.col("702_val"),
        F.col("853_id"),
        F.col("853_val")
) \

test_user_feature = spark.read.csv(
    sys.argv[4], header=False, inferSchema=False)
test_user_feature = test_user_feature.withColumn("parsed", parse_udf(F.col("_c2"))) \
    .select(
        F.col("_c0").alias("common_feature_index"),
        F.explode("parsed").alias("key_values")
) \
    .groupBy("common_feature_index") \
    .pivot("key_values.feature_name") \
    .agg(F.collect_list("key_values.feature_kv").alias("feature_kv")) \
    .withColumn("101_id", get_keys_udf(F.col("101"))) \
    .withColumn("121_id", get_keys_udf(F.col("121"))) \
    .withColumn("122_id", get_keys_udf(F.col("122"))) \
    .withColumn("124_id", get_keys_udf(F.col("124"))) \
    .withColumn("125_id", get_keys_udf(F.col("125"))) \
    .withColumn("126_id", get_keys_udf(F.col("126"))) \
    .withColumn("127_id", get_keys_udf(F.col("127"))) \
    .withColumn("128_id", get_keys_udf(F.col("128"))) \
    .withColumn("129_id", get_keys_udf(F.col("129"))) \
    .withColumn("109_14_id", get_keys_udf(F.col("109_14"))) \
    .withColumn("109_14_val", get_vals_udf(F.col("109_14"))) \
    .withColumn("110_14_id", get_keys_udf(F.col("110_14"))) \
    .withColumn("110_14_val", get_vals_udf(F.col("110_14"))) \
    .withColumn("127_14_id", get_keys_udf(F.col("127_14"))) \
    .withColumn("127_14_val", get_vals_udf(F.col("127_14"))) \
    .withColumn("150_14_id", get_keys_udf(F.col("150_14"))) \
    .withColumn("150_14_val", get_vals_udf(F.col("150_14"))) \
    .select(
        F.col("common_feature_index"),
        F.col("101_id"),
        F.col("121_id"),
        F.col("122_id"),
        F.col("124_id"),
        F.col("125_id"),
        F.col("126_id"),
        F.col("127_id"),
        F.col("128_id"),
        F.col("129_id"),
        F.col("109_14_id"),
        F.col("109_14_val"),
        F.col("110_14_id"),
        F.col("110_14_val"),
        F.col("127_14_id"),
        F.col("127_14_val"),
        F.col("150_14_id"),
        F.col("150_14_val")
)

test_sample.join(test_user_feature, on="common_feature_index", how="inner") \
    .write.mode("overwrite").saveAsTable(sys.argv[6])
