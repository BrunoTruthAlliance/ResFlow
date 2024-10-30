import tensorflow as tf

aliccp_defaults = [
    tf.int32,  # sample_id 0
    tf.int32,  # ctr_label  1
    tf.int32,  # cvr_label  2
    tf.constant(0, dtype=tf.int32),  # 205  3
    tf.constant(0, dtype=tf.int32),  # 206  4
    tf.constant(0, dtype=tf.int32),  # 207  5
    tf.constant(0, dtype=tf.int32),  # 216  6
    tf.constant(0, dtype=tf.int32),  # 301  7
    tf.constant(0, dtype=tf.int32),  # 508  8
    tf.constant(0, dtype=tf.int32),  # 509  9
    tf.constant(0, dtype=tf.int32),  # 702  10
    tf.constant(0, dtype=tf.int32),  # 101  11
    tf.constant(0, dtype=tf.int32),  # 121  12
    tf.constant(0, dtype=tf.int32),  # 122  13
    tf.constant(0, dtype=tf.int32),  # 124  14
    tf.constant(0, dtype=tf.int32),  # 125  15
    tf.constant(0, dtype=tf.int32),  # 126  16
    tf.constant(0, dtype=tf.int32),  # 127  17
    tf.constant(0, dtype=tf.int32),  # 128  18
    tf.constant(0, dtype=tf.int32),  # 129  19
    tf.constant([''], dtype=tf.string),  # 210  20
    tf.constant([''], dtype=tf.string)  # 853  21
]


feature_config = [
    {"feature_name": "205", "is_ragged": False, "row": 3, "hash_size": 3168700},
    {"feature_name": "206", "is_ragged": False, "row": 4, "hash_size": 8700},
    {"feature_name": "207", "is_ragged": False, "row": 5, "hash_size": 610400},
    {"feature_name": "216", "is_ragged": False, "row": 6, "hash_size": 210000},
    {"feature_name": "301", "is_ragged": False, "row": 7, "hash_size": 4},
    {"feature_name": "508", "is_ragged": False, "row": 8, "hash_size": 7800},
    {"feature_name": "509", "is_ragged": False, "row": 9, "hash_size": 388900},
    {"feature_name": "702", "is_ragged": False, "row": 10, "hash_size": 144000},
    {"feature_name": "101", "is_ragged": False, "row": 11,  "hash_size": 294900},
    {"feature_name": "121", "is_ragged": False, "row": 12,  "hash_size": 100},
    {"feature_name": "122", "is_ragged": False, "row": 13,  "hash_size": 16},
    {"feature_name": "124", "is_ragged": False, "row": 14,  "hash_size": 4},
    {"feature_name": "125", "is_ragged": False, "row": 15,  "hash_size": 16},
    {"feature_name": "126", "is_ragged": False, "row": 16,  "hash_size": 8},
    {"feature_name": "127", "is_ragged": False, "row": 17,  "hash_size": 8},
    {"feature_name": "128", "is_ragged": False, "row": 18,  "hash_size": 8},
    {"feature_name": "129", "is_ragged": False, "row": 19,  "hash_size": 8},
    {"feature_name": "210", "is_ragged": True, "row": 20,  "hash_size": 99000},
    {"feature_name": "853", "is_ragged": True, "row": 21,  "hash_size": 87000}
]


def preprocess_row(*row):
    features = {}
    for config in feature_config:
        if features.get(config["feature_name"]) == None:
            if config["is_ragged"]:
                features[config["feature_name"]] = tf.strings.to_number(
                    tf.strings.split(row[config["row"]], sep=';'), out_type=tf.int32)
            else:
                features[config["feature_name"]] = row[config["row"]]

    ctr_label = row[1]
    cvr_label = row[2]
    return {"features": features, "labels": {"ctr_label": ctr_label, "cvr_label": cvr_label}}
