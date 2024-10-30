import tensorflow as tf
from decoder import percentiles


class ResMultiTask():
    def __init__(self, feature_config, model_config):
        super().__init__()
        self.embedding_layers = {}
        self.embedding_dim = model_config["embedding_dim"]
        self.feature_config = feature_config
        self.target_num = model_config["target_num"]
        self.weight_increment = model_config["weight_increment"]
        self.dense1 = []
        self.dense2 = []
        self.dense3 = []

        for i in range(self.target_num):
            self.dense1.append(tf.keras.layers.Dense(model_config["dense_dim1"],
                                                     activation=tf.keras.layers.PReLU(
                                                         name='prelu1_' + str(i)),
                                                     name='dense1_' + str(i)))
            self.dense2.append(tf.keras.layers.Dense(model_config["dense_dim2"],
                                                     activation=tf.keras.layers.PReLU(
                                                         name='prelu2_' + str(i)),
                                                     name='dense2_' + str(i)))
            self.dense3.append(tf.keras.layers.Dense(model_config["dense_dim3"],
                                                     activation=None,
                                                     name='dense3_' + str(i)))

        for config in self.feature_config:
            self.embedding_layers[config["feature_name"]] = tf.keras.layers.Embedding(
                config["hash_size"], self.embedding_dim)

        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=model_config['learning_rate'], momentum=0.9)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_config['learning_rate'])

        self._is_built = False
        self._build_model()

    def _build_model(self):
        self.input_holders = {}
        for config in self.feature_config:
            if config["is_ragged"]:
                self.input_holders[config["feature_name"]+"_ragged_values"] = tf.compat.v1.placeholder(
                    tf.int32, shape=(None,), name=config["feature_name"]+"_ragged_values")
                self.input_holders[config["feature_name"]+"_ragged_row_splits"] = tf.compat.v1.placeholder(
                    tf.int64, shape=(None,), name=config["feature_name"]+"_ragged_row_splits")
                self.input_holders[config["feature_name"]] = tf.RaggedTensor.from_row_splits(
                    values=self.input_holders[config["feature_name"]+"_ragged_values"], row_splits=self.input_holders[config["feature_name"]+"_ragged_row_splits"])
            else:
                self.input_holders[config["feature_name"]] = tf.compat.v1.placeholder(
                    tf.int32, shape=(None, 1), name=config["feature_name"]+"_ph")

        for i in range(self.target_num):
            self.input_holders["label_"+str(i)] = tf.compat.v1.placeholder(
                tf.float32, shape=(None, 1), name="label_"+str(i))

        self.input_holders["score"] = tf.compat.v1.placeholder(
            tf.float32, shape=(None, 1), name="score")

        embs = []
        for config in self.feature_config:
            if config["is_ragged"]:
                emb = tf.ragged.map_flat_values(
                    self.embedding_layers[config["feature_name"]], self.input_holders[config["feature_name"]])
                emb = tf.reduce_sum(emb, axis=1)
            else:
                emb = self.embedding_layers[config["feature_name"]](
                    self.input_holders[config["feature_name"]])
                emb = tf.compat.v1.squeeze(input=emb, axis=[1])

            embs.append(emb)
            print(config["feature_name"], " shape: ", emb.shape)
        emb_out = tf.keras.layers.Concatenate(axis=1)(embs)

        logits = []
        for i in range(self.target_num):
            mid = self.dense1[i](emb_out)
            mid = self.dense2[i](mid)
            logit = self.dense3[i](mid)
            if i > 0:
                logit = tf.math.add(
                    logit, logits[i-1], name="residule_" + str(i))
            logits.append(logit)

        preds = []
        for i in range(self.target_num):
            pred = tf.math.sigmoid(logits[i], name="sigmoid" + str(i))
            pred = tf.clip_by_value(
                pred, clip_value_min=1e-8, clip_value_max=1-1e-8)
            preds.append(pred)

        self.final_loss = 0.0

        self.auc = []
        self.auc_op = []
        pos_weights = [0.9 + self.weight_increment *
                       i for i in range(self.target_num)]
        for i in range(self.target_num):
            loss = tf.compat.v1.nn.weighted_cross_entropy_with_logits(
                labels=self.input_holders["label_"+str(i)], logits=logits[i], pos_weight=pos_weights[i])
            loss = tf.reduce_mean(loss)
            self.final_loss = self.final_loss + loss

            auc, auc_op = tf.compat.v1.metrics.auc(
                labels=self.input_holders["label_"+str(i)], predictions=preds[i])
            self.auc.append(auc)
            self.auc_op.append(auc_op)

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_op = self.optimizer.minimize(
            loss=self.final_loss, global_step=self.global_step)
        score = preds[self.target_num-1] * 93
        for i in range(self.target_num-2, -1, -1):
            p = tf.maximum(preds[i] - preds[i+1], tf.zeros_like(preds[i]))
            score = score + p * (percentiles[i + 1] + percentiles[i]) / 2

        score = score + (1 - preds[0]) * 10 / 2
        # score = tf.minimum(score, self.input_holders[""])
        self.MSE = tf.reduce_mean(
            tf.square(score - self.input_holders["score"]))
        self.avg_score = tf.reduce_mean(score)

        self._is_built = True

    def forward(self, batch, mode, session):
        if not self._is_built:
            raise ValueError(
                "Model graph is not built. Call _build_model() in the constructor.")

        feed_dict = {}
        features = batch["features"]
        labels = batch["labels"]
        scores = batch["score"]
        for config in self.feature_config:
            feature_name = config["feature_name"]
            if config["is_ragged"]:
                ph = self.input_holders[feature_name]
                feed_dict[ph] = features[feature_name]
            else:
                ph = self.input_holders[feature_name]
                feed_dict[ph] = features[feature_name].reshape(-1, 1)

        for i in range(self.target_num):
            feed_dict[self.input_holders["label_" +
                                         str(i)]] = labels["label_"+str(i)].reshape(-1, 1)

        feed_dict[self.input_holders["score"]] = scores.reshape(-1, 1)

        if mode == "train":
            # Training mode
            _, avg_score, loss_value, mse_value, step, a1, a2, a3, a4, a5, a6, a7, a8, a9, _, _, _, _, _, _, _, _, _ = session.run(     # a11, a12, a13, a14, a15, a16, a17, a18, a19, a20,
                [self.train_op, self.avg_score, self.final_loss,
                    self.MSE, self.global_step] + self.auc + self.auc_op,
                feed_dict=feed_dict
            )
            return avg_score, loss_value, mse_value, step, a1, a2, a3, a4, a5, a6, a7, a8, a9
        elif mode == "test":
            # Testing mode
            avg_score, mse_value, a1, a2, a3, a4, a5, a6, a7, a8, a9, _, _, _, _, _, _, _, _, _ = session.run(     # a11, a12, a13, a14, a15, a16, a17, a18, a19, a20,
                [self.avg_score, self.MSE] + self.auc + self.auc_op,
                feed_dict=feed_dict
            )
            return avg_score, mse_value, a1, a2, a3, a4, a5, a6, a7, a8, a9
        else:
            raise ValueError(f"Unsupported mode: {mode}")
