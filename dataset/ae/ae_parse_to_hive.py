from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import sys


spark = SparkSession.builder.appName(
    'parse csv ori data to table').enableHiveSupport().getOrCreate()

user_feature = spark.read.csv(sys.argv[1], header=False, inferSchema=False) \
    .select(
        F.col("_c0").alias("pv_id"),
        F.col("_c1").alias("u1"),
        F.col("_c2").alias("u2"),
        F.col("_c3").alias("u3"),
        F.col("_c4").alias("u4"),
        F.col("_c5").alias("u5"),
        F.col("_c6").alias("u6"),
        F.col("_c7").alias("u7"),
        F.concat_ws(";", F.array(F.col("_c8"), F.col(
            "_c9"), F.col("_c10"))).alias("mu1"),
        F.concat_ws(";", F.array(F.col("_c11"), F.col("_c12"), F.col(
            "_c13"), F.col("_c14"), F.col("_c15"))).alias("mu2"),
        F.concat_ws(";", F.array(F.col("_c16"), F.col("_c17"), F.col("_c18"), F.col("_c19"), F.col(
            "_c20"), F.col("_c21"), F.col("_c22"), F.col("_c23"), F.col("_c24"), F.col("_c25"))).alias("mu3"),
        F.concat_ws(";", F.array(F.col("_c26"), F.col(
            "_c27"), F.col("_c28"))).alias("mu4"),
        F.col("_c29").alias("u8"),
        F.col("_c30").alias("u9"),
        F.col("_c31").alias("u10"),
        F.col("_c32").alias("u11")
)

item_feature_samples = spark.read.csv(sys.argv[2], header=False, inferSchema=False) \
    .select(
        F.col("_c0").alias("pv_id"),
        F.col("_c1").alias("i1"),
        F.col("_c2").alias("i2"),
        F.col("_c3").alias("i3"),
        F.col("_c4").alias("i4"),
        F.col("_c5").alias("i5"),
        F.col("_c6").alias("i6"),
        F.col("_c7").alias("i7"),
        F.col("_c8").alias("i8"),
        F.col("_c9").alias("i9"),
        F.col("_c10").alias("i10"),
        F.col("_c11").alias("i11"),
        F.col("_c12").alias("i12"),
        F.col("_c13").alias("i13"),
        F.col("_c14").alias("i14"),
        F.col("_c15").alias("i15"),
        F.col("_c16").alias("i16"),
        F.col("_c17").alias("i17"),
        F.col("_c18").alias("i18"),
        F.col("_c19").alias("i19"),
        F.col("_c20").alias("i20"),
        F.col("_c21").alias("i21"),
        F.col("_c22").alias("i22"),
        F.col("_c23").alias("i23"),
        F.col("_c24").alias("i24"),
        F.col("_c25").alias("i25"),
        F.col("_c26").alias("i26"),
        F.col("_c27").alias("i27"),
        F.col("_c28").alias("i28"),
        F.col("_c29").alias("i29"),
        F.col("_c30").alias("i30"),
        F.col("_c31").alias("i31"),
        F.col("_c32").alias("i32"),
        F.col("_c33").alias("i33"),
        F.col("_c34").alias("i34"),
        F.col("_c35").alias("i35"),
        F.col("_c36").alias("i36"),
        F.col("_c37").alias("i37"),
        F.col("_c38").alias("i38"),
        F.col("_c39").alias("i39"),
        F.col("_c40").alias("i40"),
        F.col("_c41").alias("i41"),
        F.col("_c42").alias("i42"),
        F.col("_c43").alias("i43"),
        F.col("_c44").alias("i44"),
        F.col("_c45").alias("i45"),
        F.col("_c46").alias("i46"),
        F.col("_c47").alias("i47"),
        F.col("_c48").alias("label")
)

item_feature_samples.join(user_feature, on="pv_id", how="inner") \
    .write.mode("overwrite").saveAsTable(sys.argv[3])
