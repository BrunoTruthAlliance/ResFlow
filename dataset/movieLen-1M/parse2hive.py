from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName(
    'parse txt ori data to table').enableHiveSupport().getOrCreate()
movies = spark.read.text(sys.argv[1]) \
    .select(F.split(F.col("value"), "::").alias("rows")) \
    .select(
        F.col("rows")[0].alias('movie_id'),
        F.col("rows")[1].alias('movie_title'),
        F.col("rows")[2].alias('movie_genre')
)

ratings = spark.read.text(sys.argv[2]) \
    .select(F.split(F.col("value"), "::").alias("rows")) \
    .select(
        F.col("rows")[0].alias('user_id'),
        F.col("rows")[1].alias('movie_id'),
        F.col("rows")[2].alias('rating'),
        F.from_unixtime(F.col("rows")[3], "yyyy-MM-dd HH:mm:ss").alias('time'),
        F.from_unixtime(F.col("rows")[3], "yyyy-MM-dd").alias('date'),
)

users = spark.read.text(sys.argv[3]) \
    .select(F.split(F.col("value"), "::").alias("rows")) \
    .select(
        F.col("rows")[0].alias('user_id'),
        F.col("rows")[1].alias('gender'),
        F.col("rows")[2].alias('age'),
        F.col("rows")[3].alias('job'),
        F.col("rows")[4].alias('loc')
)

samples = ratings.join(movies, on="movie_id", how="left") \
    .join(users, on="user_id", how="left") \
    .write.mode("overwrite").saveAsTable(sys.argv[4])
