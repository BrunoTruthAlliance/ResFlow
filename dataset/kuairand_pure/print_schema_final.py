from os import path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName(
    "Print schema").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
spark.read.parquet(sys.argv[1]).printSchema()
