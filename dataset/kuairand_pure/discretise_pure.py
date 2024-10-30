from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.feature import Bucketizer
import sys

spark = SparkSession.builder.appName(
    'parse csv ori data to table').enableHiveSupport().getOrCreate()


def filter_zeros(array):
    # In case boundary is 0
    res = [float(x) for x in array if x > 0.0]
    if len(res) == 0:
        return [1.0]
    else:
        res = sorted(set(res))
        return res


filter_zeros_udf = F.udf(filter_zeros, ArrayType(DoubleType()))

df = spark.table(sys.argv[1])
dt = df.dtypes
dense_columns = [item[0] for item in dt if item[1].startswith('double')]
numeric_int_columns = [
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    "counts",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time"
]
bucket_columns = dense_columns + numeric_int_columns
cols = [filter_zeros_udf(F.col(f"percent_{c}")).alias(
    f"{c}_b") for c in bucket_columns]

boundaries = spark.read.parquet(sys.argv[2]) \
    .select(*cols) \
    .collect()

boundaries = boundaries[0].asDict()
inputCols = []
outputCols = []
splitsArray = []
for column_name, column_value in boundaries.items():
    inputCol = column_name.rstrip("_b")
    outputCol = "id_" + inputCol
    splitArray = [-float("inf")] + column_value + [float("inf")]
    print(
        f"inputCol: {inputCol}, outputCol: {outputCol}, splitArray: {splitArray}")
    inputCols.append(inputCol)
    outputCols.append(outputCol)
    splitsArray.append(splitArray)


df_indexed = spark.read.parquet(sys.argv[3])
column_names = df_indexed.columns
ori_cols = [F.col(c).cast("double").alias(
    c) if c in numeric_int_columns else F.col(c) for c in column_names]
ori_features = df_indexed.select(*ori_cols)


bucketiser = Bucketizer(splitsArray=splitsArray,
                        inputCols=inputCols, outputCols=outputCols)
bucketed2 = bucketiser.setHandleInvalid("keep").transform(ori_features)

bucketed2.write.mode("overwrite").parquet(sys.argv[4])
