from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName(
    'paper data hive to csv').enableHiveSupport().getOrCreate()
num_partitions = int(sys.argv[3])
df = spark.read.table(sys.argv[1]) \
    .filter(F.col("301") == int(sys.argv[4])) \
    .select(
        F.col("sample_id"),
        F.col("ctr_label"),
        F.col("cvr_label"),
        F.col("205"),
        F.col("206"),
        F.col("207"),
        F.col("216"),
        F.col("301"),
        F.col("508"),
        F.col("509"),
        F.col("702"),
        F.col("101"),
        F.col("121"),
        F.col("122"),
        F.col("124"),
        F.col("125"),
        F.col("126"),
        F.col("127"),
        F.col("128"),
        F.col("129"),
        F.concat_ws(";", F.col("210")).alias("210"),
        F.concat_ws(";", F.col("853")).alias("853")
) \
    .orderBy(F.rand()) \
    .repartition(num_partitions) \
    .write.mode('overwrite').option("header", True).csv(sys.argv[2])
