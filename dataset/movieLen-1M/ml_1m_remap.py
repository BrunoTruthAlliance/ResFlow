from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
import sys

spark = SparkSession.builder.appName(
    'process the feature and sample data').enableHiveSupport().getOrCreate()

samples = spark.read.table(sys.argv[1]) \
    .select(
        F.col("user_id"),
        F.col("movie_id"),
        F.col("rating"),
        F.split(F.col("movie_genre"), "\|").alias("movie_genres"),
        F.col("gender"),
        F.col("age"),
        F.col("job"),
        F.col("loc"),
        F.when(F.col("date") > "2000-12-02", 0).otherwise(1).alias("is_train")
)

column_set = ["gender", "age", "job", "loc"]

for column in column_set:
    string_indexer = StringIndexer(
        inputCol=column, outputCol=column+"_id", handleInvalid="keep")
    string_indexer_model = string_indexer.fit(samples)
    samples = string_indexer_model.transform(samples)

movie_genres_map_df = samples.select(F.col("user_id"), F.col(
    "movie_id"), F.explode(F.col("movie_genres")).alias("genre"))
string_indexer = StringIndexer(
    inputCol="genre", outputCol="genre_id", handleInvalid="keep")
string_indexer_model = string_indexer.fit(movie_genres_map_df)
movie_genres_map_df = string_indexer_model.transform(movie_genres_map_df) \
    .select(
    F.col("user_id"),
    F.col("movie_id"),
    F.col("genre_id").cast(
        "bigint")
) \
    .groupBy("user_id", "movie_id") \
    .agg(F.collect_list("genre_id").alias("genre_ids"))

train_sample = samples.join(movie_genres_map_df, on=["user_id", "movie_id"], how="inner") \
    .select(
    F.col("user_id"),
    F.col("movie_id"),
    F.col("rating"),
    F.col("movie_genres"),
    F.col("is_train"),
    F.col("gender"),
    F.col("age"),
    F.col("job"),
    F.col("loc"),
    F.col("gender_id").cast("bigint").alias("gender_id"),
    F.col("age_id").cast("bigint").alias("age_id"),
    F.col("job_id").cast("bigint").alias("job_id"),
    F.col("loc_id").cast("bigint").alias("loc_id"),
    F.col("genre_ids")
) \
    .write.mode("overwrite").saveAsTable(sys.argv[2])
