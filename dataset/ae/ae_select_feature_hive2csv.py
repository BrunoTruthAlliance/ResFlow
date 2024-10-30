from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys

spark = SparkSession.builder.appName(
    'paper data hive to csv').enableHiveSupport().getOrCreate()
num_partitions = int(sys.argv[3])
feature = spark.read.table(sys.argv[1])
column_names = ['pv_id', 'label', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'mu1', 'mu2', 'mu3', 'mu4', 'u8_id', 'u9_id', 'u10', 'u11', 'i1_id', 'i2_id', 'i3_id', 'i4_id', 'i5_id', 'i6_id', 'i7_id', 'i8_id', 'i9_id', 'i10_id', 'i11_id', 'i12_id', 'i13_id', 'i14_id', 'i15_id', 'i16_id',
                'i17_id', 'i18', 'i19_id', 'i20_id', 'i21_id', 'i22_id', 'i23_id', 'i24_id', 'i25_id', 'i26_id', 'i27_id', 'i28_id', 'i29_id', 'i30_id', 'i31_id', 'i32_id', 'i33', 'i34', 'i35', 'i36_id', 'i37_id', 'i38_id', 'i39', 'i40', 'i41', 'i42_id', 'i43_id', 'i44_id', 'i45_id', 'i46_id', 'i47_id']
cols = [F.col(cn) if not (cn.startswith("i") or cn == "u8_id" or cn == "u9_id")else F.col(
    cn).cast("bigint").alias(cn.split("_")[0]) for cn in column_names]
feature = feature.select(*cols) \
    .orderBy(F.rand()) \
    .repartition(num_partitions) \
    .write.mode('overwrite').option("header", True).csv(sys.argv[2])
