from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys

need_indexed = [
    # String
    "user_active_degree",
    "user_follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "video_type",
    "upload_type",
    "tag",
    # Int
    "is_live_streamer",
    "user_id",
    "video_id",
    "author_id",
    "music_id"
]


spark = SparkSession.builder.appName(
    'Index string columns of kuaiRand').enableHiveSupport().getOrCreate()
df = spark.table(sys.argv[1])
for c in need_indexed:
    df2 = df.select(c).dropna().distinct()
    df2 = df2.select(c, (F.row_number().over(
        Window.orderBy(c)) - F.lit(1)).alias(f"index_{c}"))
    df2 = F.broadcast(df2)
    df = df.join(df2, [c], "left")
df.write.mode('overwrite').parquet(sys.argv[2])
