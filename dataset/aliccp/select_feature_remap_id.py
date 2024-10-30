from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
from pyspark.ml.feature import StringIndexer
import sys

spark = SparkSession.builder.appName(
    'remap id column').enableHiveSupport().getOrCreate()

train_sample = spark.read.table(sys.argv[1]) \
    .select(
        F.col("sample_id"),
        F.col("ctr_label"),
        F.col("cvr_label"),
        F.col("common_feature_index"),
        F.col("205_id"),
        F.col("206_id"),
        F.col("207_id"),
        F.col("216_id"),
        F.col("301_id"),
        F.col("508_id"),
        F.col("509_id"),
        F.col("702_id"),
        F.col("101_id"),
        F.col("121_id"),
        F.col("122_id"),
        F.col("124_id"),
        F.col("125_id"),
        F.col("126_id"),
        F.col("127_id"),
        F.col("128_id"),
        F.col("129_id"),
        F.split(F.col("853_id"), ';').alias("853_id"),
        F.split(F.col("210_id"), ';').alias("210_id")
)
test_sample = spark.read.table(sys.argv[2]) \
    .select(
        F.col("sample_id"),
        F.col("ctr_label"),
        F.col("cvr_label"),
        F.col("common_feature_index"),
        F.col("205_id"),
        F.col("206_id"),
        F.col("207_id"),
        F.col("216_id"),
        F.col("301_id"),
        F.col("508_id"),
        F.col("509_id"),
        F.col("702_id"),
        F.col("101_id"),
        F.col("121_id"),
        F.col("122_id"),
        F.col("124_id"),
        F.col("125_id"),
        F.col("126_id"),
        F.col("127_id"),
        F.col("128_id"),
        F.col("129_id"),
        F.split(F.col("853_id"), ';').alias("853_id"),
        F.split(F.col("210_id"), ';').alias("210_id")
)

column_set1 = ["205", "206", "207", "216", "301", "508", "509", "702",
               "101", "121", "122", "124", "125", "126", "127", "128", "129"]
column_set2 = ["210", "853"]

for column in column_set1:
    string_indexer = StringIndexer(
        inputCol=column+"_id", outputCol=column+"_reid", handleInvalid="keep")
    string_indexer_model = string_indexer.fit(train_sample)
    train_sample = string_indexer_model.transform(train_sample)
    test_sample = string_indexer_model.transform(test_sample)

for column in column_set2:
    train_mapping_df = train_sample.select(
        F.col("sample_id"), F.explode(F.col(column+"_id")).alias(column+"_eid"))
    test_mapping_df = test_sample.select(
        F.col("sample_id"), F.explode(F.col(column+"_id")).alias(column+"_eid"))
    string_indexer = StringIndexer(
        inputCol=column+"_eid", outputCol=column+"_reid", handleInvalid="keep")
    string_indexer_model = string_indexer.fit(train_mapping_df)
    train_mapping_df = string_indexer_model.transform(train_mapping_df) \
        .select(
        F.col("sample_id"),
        F.col(
            column+"_reid").cast("bigint")
    ) \
        .groupBy("sample_id") \
        .agg(F.collect_list(column+"_reid").alias(column+"_reids"))
    train_sample = train_sample.join(
        train_mapping_df, on="sample_id", how="inner")
    test_mapping_df = string_indexer_model.transform(test_mapping_df) \
        .select(
        F.col("sample_id"),
        F.col(
            column+"_reid").cast("bigint")
    ) \
        .groupBy("sample_id") \
        .agg(F.collect_list(column+"_reid").alias(column+"_reids"))
    test_sample = test_sample.join(
        test_mapping_df, on="sample_id", how="inner")

train_sample \
    .select(
        F.col("sample_id").cast("bigint"),
        F.col("ctr_label").cast("bigint"),
        F.col("cvr_label").cast("bigint"),
        F.col("common_feature_index"),
        F.col("205_reid").cast("bigint").alias("205"),
        F.col("206_reid").cast("bigint").alias("206"),
        F.col("207_reid").cast("bigint").alias("207"),
        F.col("216_reid").cast("bigint").alias("216"),
        F.col("301_reid").cast("bigint").alias("301"),
        F.col("508_reid").cast("bigint").alias("508"),
        F.col("509_reid").cast("bigint").alias("509"),
        F.col("702_reid").cast("bigint").alias("702"),
        F.col("101_reid").cast("bigint").alias("101"),
        F.col("121_reid").cast("bigint").alias("121"),
        F.col("122_reid").cast("bigint").alias("122"),
        F.col("124_reid").cast("bigint").alias("124"),
        F.col("125_reid").cast("bigint").alias("125"),
        F.col("126_reid").cast("bigint").alias("126"),
        F.col("127_reid").cast("bigint").alias("127"),
        F.col("128_reid").cast("bigint").alias("128"),
        F.col("129_reid").cast("bigint").alias("129"),
        F.col("210_reids").alias("210"),
        F.col("853_reids").alias("853")
    ) \
    .write.mode("overwrite").saveAsTable(sys.argv[3])
test_sample \
    .select(
        F.col("sample_id").cast("bigint"),
        F.col("ctr_label").cast("bigint"),
        F.col("cvr_label").cast("bigint"),
        F.col("common_feature_index"),
        F.col("205_reid").cast("bigint").alias("205"),
        F.col("206_reid").cast("bigint").alias("206"),
        F.col("207_reid").cast("bigint").alias("207"),
        F.col("216_reid").cast("bigint").alias("216"),
        F.col("301_reid").cast("bigint").alias("301"),
        F.col("508_reid").cast("bigint").alias("508"),
        F.col("509_reid").cast("bigint").alias("509"),
        F.col("702_reid").cast("bigint").alias("702"),
        F.col("101_reid").cast("bigint").alias("101"),
        F.col("121_reid").cast("bigint").alias("121"),
        F.col("122_reid").cast("bigint").alias("122"),
        F.col("124_reid").cast("bigint").alias("124"),
        F.col("125_reid").cast("bigint").alias("125"),
        F.col("126_reid").cast("bigint").alias("126"),
        F.col("127_reid").cast("bigint").alias("127"),
        F.col("128_reid").cast("bigint").alias("128"),
        F.col("129_reid").cast("bigint").alias("129"),
        F.col("210_reids").alias("210"),
        F.col("853_reids").alias("853")
    ) \
    .write.mode("overwrite").saveAsTable(sys.argv[4])

# train_mapping_df.filter(F.col(column+"_eid") != "").select(F.col(column+"_eid"), F.col(column+"_reid")).distinct().orderBy(F.col(column+"_reid")).show(truncate=False)
# test_mapping_df.filter(F.col(column+"_eid") != "").select(F.col(column+"_eid"), F.col(column+"_reid")).distinct().orderBy(F.col(column+"_reid")).show(truncate=False)
