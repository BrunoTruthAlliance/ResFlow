from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import sys

spark = SparkSession.builder.appName(
    'get percentile').enableHiveSupport().getOrCreate()

percentiles = [(1.0+i)/100 for i in range(0, 99)]
percentiles_str = "array("+",".join([str(num) for num in percentiles])+")"

cols = []

for i in range(1, 48):
    tmp = "percentile_approx(i" + str(i) + ", " + \
        percentiles_str + ", 10000) as percent" + "_i" + str(i)
    cols.append(tmp)

tmp = "percentile_approx(u8, " + percentiles_str + \
    ", 10000) as percent" + "_u8"
cols.append(tmp)
tmp = "percentile_approx(u9, " + percentiles_str + \
    ", 10000) as percent" + "_u9"
cols.append(tmp)

cols_str = ",".join(cols)

sql_str = "SELECT " + cols_str + " FROM " + sys.argv[1]
print(sql_str)

spark.sql(sql_str).write.mode("overwrite").saveAsTable(sys.argv[2])
