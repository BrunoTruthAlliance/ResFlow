from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.feature import Bucketizer
import sys

spark = SparkSession.builder.appName(
    'parse csv ori data to table').enableHiveSupport().getOrCreate()


def filter_zeros(array):
    res = [x for x in array if x > 0.0]
    if len(res) == 0:
        return [1.0]
    else:
        res = sorted(set(res))
        return res


filter_zeros_udf = F.udf(filter_zeros, ArrayType(DoubleType()))

cols = [filter_zeros_udf(F.col("percent_i"+str(i))
                         ).alias("i"+str(i)+"_b") for i in range(1, 48)]
cols.append(filter_zeros_udf(F.col("percent_u8")).alias("u8_b"))
cols.append(filter_zeros_udf(F.col("percent_u9")).alias("u9_b"))

boundaries = spark.read.table(sys.argv[1]) \
    .select(*cols) \
    .collect()

boundaries = boundaries[0].asDict()
inputCols = []
outputCols = []
splitsArray = []
for column_name, column_value in boundaries.items():
    inputCol = column_name.split("_")[0]
    outputCol = column_name.split("_")[0] + "_id"
    splitArray = [-float("inf")] + column_value + [float("inf")]
    print(
        f"inputCol: {inputCol}, outputCol: {outputCol}, splitArray: {splitArray}")
    inputCols.append(inputCol)
    outputCols.append(outputCol)
    splitsArray.append(splitArray)


ori_features = spark.read.table(sys.argv[2])
column_names = ori_features.columns
ori_cols = [F.col(n) if not (n.startswith("i") or n == "u8" or n == "u9")
            else F.col(n).cast("double").alias(n) for n in column_names]
ori_features = ori_features.select(*ori_cols)


bucketizer = Bucketizer(splitsArray=splitsArray,
                        inputCols=inputCols, outputCols=outputCols)
bucketed2 = bucketizer.setHandleInvalid("keep").transform(ori_features)

bucketed2.write.mode("overwrite").saveAsTable(sys.argv[3])
