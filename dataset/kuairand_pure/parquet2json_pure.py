from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.appName(
    'convert parquet to json').enableHiveSupport().getOrCreate()
df = spark.read.parquet(sys.argv[1])
df.write.mode('overwrite').json(sys.argv[2])
