from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.appName(
    'Get buckets of numeric columns kuaiRand pure').enableHiveSupport().getOrCreate()

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

percentiles = [(1.0+i)/100 for i in range(0, 99)]
percentiles_str = "array("+",".join([str(num) for num in percentiles])+")"

cols = []

for c in bucket_columns:
    tmp = f"percentile_approx({c}, {percentiles_str}, 10000) as percent_{c}"
    cols.append(tmp)

cols_str = ",".join(cols)

sql_str = f"SELECT {cols_str} FROM dev_search_algo.kuairand_pure"
print(sql_str)

spark.sql(sql_str).write.mode("overwrite").parquet(sys.argv[2])
