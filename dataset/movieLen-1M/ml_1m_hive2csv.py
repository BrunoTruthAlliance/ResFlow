from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys

spark = SparkSession.builder.appName(
    'process the feature and sample data').enableHiveSupport().getOrCreate()
samples = spark.read.table(sys.argv[1]) \
    .select(
        F.col("user_id"),
        F.col("movie_id"),
        F.col("rating"),
        F.col("is_train"),
        F.col("gender_id"),
        F.col("age_id"),
        F.col("job_id"),
        F.col("loc_id"),
        F.concat_ws(";", F.col("genre_ids")).alias("genre_ids")
)
# train sample
samples.filter(F.col("is_train") == 1) \
    .orderBy(F.rand()) \
    .repartition(4) \
    .write.mode('overwrite').option("header", True).csv(sys.argv[2])

# test sample
samples.filter(F.col("is_train") == 0) \
    .orderBy(F.rand()) \
    .repartition(4) \
    .write.mode('overwrite').option("header", True).csv(sys.argv[3])
