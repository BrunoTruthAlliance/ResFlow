import tensorflow as tf


class ResMultiTask():
    def __init__(self, feature_config, model_config):
        super().__init__()
        self.embedding_layers = {}
        self.embedding_dim = model_config["embedding_dim"]
        self.feature_config = feature_config

        self.emb_bn = tf.keras.layers.BatchNormalization()

        self.ctr_dense1 = tf.keras.layers.Dense(model_config["dense_dim1"], activation=tf.keras.layers.PReLU(name='ctr_prelu1'),
                                                name='ctr_dense1')
        self.ctr_dense2 = tf.keras.layers.Dense(model_config["dense_dim2"], activation=tf.keras.layers.PReLU(name='ctr_prelu2'),
                                                name='ctr_dense2')
        self.ctr_dense3 = tf.keras.layers.Dense(model_config["dense_dim3"],
                                                name='ctr_dense3')

        self.ctr_bn1 = tf.keras.layers.BatchNormalization()
        self.ctr_bn2 = tf.keras.layers.BatchNormalization()

        self.cvr_dense1 = tf.keras.layers.Dense(model_config["dense_dim1"], activation=tf.keras.layers.PReLU(name='cvr_prelu1'),
                                                name='cvr_dense1')
        self.cvr_dense2 = tf.keras.layers.Dense(model_config["dense_dim2"], activation=tf.keras.layers.PReLU(name='cvr_prelu2'),
                                                name='cvr_dense2')
        self.cvr_dense3 = tf.keras.layers.Dense(model_config["dense_dim3"],
                                                name='cvr_dense3')

        self.cvr_bn1 = tf.keras.layers.BatchNormalization()
        self.cvr_bn2 = tf.keras.layers.BatchNormalization()

        for config in self.feature_config:
            self.embedding_layers[config["feature_name"]] = tf.keras.layers.Embedding(
                config["hash_size"], self.embedding_dim)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=model_config['learning_rate'])

        self.loss_ctr = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, name='bce_ctr')
        self.loss_cvr = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, name='bce_cvr')

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

        self.input_holders["ctr_label"] = tf.compat.v1.placeholder(
            tf.float32, shape=(None, 1), name="ctr_label_ph")
        self.input_holders["cvr_label"] = tf.compat.v1.placeholder(
            tf.float32, shape=(None, 1), name="cvr_label_ph")

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

        emb_out = self.emb_bn(emb_out)

        ctr_out1 = self.ctr_dense1(emb_out)
        ctr_out1 = self.ctr_bn1(ctr_out1)
        ctr_out2 = self.ctr_dense2(ctr_out1)
        ctr_out2 = self.ctr_bn2(ctr_out2)
        ctr_logit = self.ctr_dense3(ctr_out2)
        ctr_pred = tf.math.sigmoid(ctr_logit, name="ctr_sigmoid")
        ctr_pred = tf.clip_by_value(
            ctr_pred, clip_value_min=1e-8, clip_value_max=1-1e-8)

        cvr_out1 = self.cvr_dense1(emb_out)
        cvr_out1 = self.cvr_bn1(cvr_out1)
        cvr_out2 = self.cvr_dense2(tf.math.add(
            ctr_out1, cvr_out1, name="residule1"))
        cvr_out2 = self.cvr_bn2(cvr_out2)
        cvr_logit = self.cvr_dense3(tf.math.add(
            ctr_out2, cvr_out2, name="residule2"))
        cvr_logit = tf.math.add(ctr_logit, cvr_logit, name="residule3")
        cvr_pred = tf.math.sigmoid(cvr_logit, name="cvr_sigmoid")
        cvr_pred = tf.clip_by_value(
            cvr_pred, clip_value_min=1e-8, clip_value_max=1-1e-8)

        # Notion: The positive sample weight of each task is only for Ali-CCP, and the others weights can be found in the paper Appendix A
        ctr_loss = tf.compat.v1.nn.weighted_cross_entropy_with_logits(
            labels=self.input_holders["ctr_label"], logits=ctr_logit, pos_weight=100)
        cvr_loss = tf.compat.v1.nn.weighted_cross_entropy_with_logits(
            labels=self.input_holders["cvr_label"], logits=cvr_logit, pos_weight=500)
        self.final_loss = 2.0 * \
            tf.reduce_mean(ctr_loss) + 1.0 * tf.reduce_mean(cvr_loss)

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.train_op = self.optimizer.minimize(
            loss=self.final_loss, global_step=self.global_step)
        self.auc_ctr, self.auc_op_ctr = tf.compat.v1.metrics.auc(
            labels=self.input_holders["ctr_label"], predictions=ctr_pred)
        self.auc_cvr, self.auc_op_cvr = tf.compat.v1.metrics.auc(
            labels=self.input_holders["cvr_label"], predictions=cvr_pred)
        self._is_built = True

    def forward(self, batch, mode, session):
        if not self._is_built:
            raise ValueError(
                "Model graph is not built. Call _build_model() in the constructor.")

        feed_dict = {}
        features = batch["features"]
        labels = batch["labels"]
        for config in self.feature_config:
            feature_name = config["feature_name"]

            if config["is_ragged"]:
                ragged_values_ph = self.input_holders[feature_name +
                                                      "_ragged_values"]
                ragged_row_splits_ph = self.input_holders[feature_name +
                                                          "_ragged_row_splits"]
                feed_dict[ragged_values_ph] = features[feature_name].values
                feed_dict[ragged_row_splits_ph] = features[feature_name].row_splits
            else:
                ph = self.input_holders[feature_name]
                feed_dict[ph] = features[feature_name].reshape(-1, 1)
        feed_dict[self.input_holders["ctr_label"]
                  ] = labels["ctr_label"].reshape(-1, 1)
        feed_dict[self.input_holders["cvr_label"]
                  ] = labels["cvr_label"].reshape(-1, 1)

        if mode == "train":
            # Training mode
            _, loss_value, auc_ctr_value, _, auc_cvr_value, _, step = session.run(
                [self.train_op, self.final_loss, self.auc_ctr, self.auc_op_ctr,
                    self.auc_cvr, self.auc_op_cvr, self.global_step],
                feed_dict=feed_dict
            )
            return loss_value, auc_ctr_value, auc_cvr_value, step
        elif mode == "test":
            # Testing mode
            auc_ctr_value, _, auc_cvr_value, _, = session.run(
                [self.auc_ctr, self.auc_op_ctr, self.auc_cvr, self.auc_op_cvr],
                feed_dict=feed_dict
            )
            return auc_ctr_value, auc_cvr_value
        else:
            raise ValueError(f"Unsupported mode: {mode}")
