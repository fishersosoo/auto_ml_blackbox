# coding=utf-8
import tensorflow as tf

from utils.controller import build_controller_model
from utils.dag import build_DAG_model


class DAGModel(tf.keras.Model):
    def __init__(self,
                 max_len,
                 input_dim,
                 num_node,
                 model_hidden_size,
                 num_class,
                 link_embedding_dim,
                 link_hidden_size,
                 M
                 ):
        """

        Args:
            model: DAG模型
            controller: controller模型
            class_num: 类别数量
            node_num: 节点数量
        """
        super(DAGModel, self).__init__()
        self.model = build_DAG_model(num_node, max_len, input_dim, num_class, model_hidden_size)
        self.controller = build_controller_model(num_node, link_embedding_dim, link_hidden_size)
        self.M = M
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.g_loss = tf.losses.categorical_crossentropy

    def train_step(self, data):
        x, y = data
        input_embeddings_train = x["train"]["input_embeddings"]
        input_mask_train = x["train"]["input_mask"]
        label_train = y["train"]["label"]
        prob_train = y["train"]["prob"]
        batch_size = x["train"]["input_embeddings"].shape[0]

        # 使用控制器创建M个子模型
        all_link, link_ce_loss = self.controller({"input": tf.constant(tf.zeros[self.M, 1])})
