# coding=utf-8
"""
ENAS算法控制器模型
"""
import tensorflow as tf
import numpy as np


class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(AdditiveAttention, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w_1 = tf.keras.layers.Dense(input_shape[-1], use_bias=False, activation=None)
        self.w_2 = tf.keras.layers.Dense(input_shape[-1], use_bias=False, activation=None)
        self.w_a = tf.keras.layers.Dense(1, use_bias=False, activation=tf.nn.sigmoid)

    def call(self, inputs, **kwargs):
        """

        Args:
            inputs: [key, query],key: [b, j-1, d], query: [b,1, d]
            **kwargs:

        Returns:
            # [b, j-1,1]
        """
        key, query = inputs
        score = tf.keras.activations.tanh(self.w_1(query) + self.w_2(key))  # [b, j-1, d]
        score = self.w_a(score)  # [b, j-1, 1]
        return score


def build_controller_model(num_node, link_embedding_dim, hidden_size):
    """

    Args:
        type_embedding_dim: 节点类型嵌入维度
        link_embedding_dim: 连接嵌入维度
        num_node:节点总数
        num_type:节点类型数量
        hidden_size: LSTM输出维度

    Returns:
        controller模型，
        返回两个状态矩阵
        [b,num_node,num_node]的连接矩阵
        和[num_node, num_type]的类型矩阵
    """

    input_tensor = tf.keras.Input(shape=[1], name="input")  # [B,1]
    batch_size = 1
    link_embedding_layer = tf.keras.layers.Dense(link_embedding_dim, use_bias=False, name="link_embedding")
    link_lstm_layer = tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True,
                                           recurrent_activation=None, name="link_lstm")
    init_link_input = tf.Variable(initial_value=tf.initializers.glorot_uniform()([1, link_embedding_dim]),
                                  shape=[1, link_embedding_dim], trainable=True, name="init_link_inputs")
    # 加性注意力
    link_atten_w_1 = tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None)
    link_atten_w_2 = tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None)
    link_atten_w_a = tf.keras.layers.Dense(1, use_bias=False, activation=tf.nn.sigmoid)

    inputs = init_link_input
    inputs = tf.broadcast_to(inputs, shape=[batch_size, link_embedding_dim])  # [B, link_embedding_dim]
    link_state = None
    all_link_outputs = [inputs]  # 连接向量lstm层的输出 [j, B, link_embedding_dim]
    inputs = tf.reshape(inputs, [batch_size, 1, link_embedding_dim])
    all_links = [tf.zeros(shape=[batch_size, num_node])]  # 生成的连接向量

    all_link_ce_loss = []  # [num_node, B]
    for j in range(2, num_node + 1):
        link_o, link_c, link_h = link_lstm_layer(inputs, initial_state=link_state)  # [B, link_embedding_dim]
        link_state = [link_c, link_h]
        all_link_outputs.append(link_o)  # [j, B, link_embedding_dim]
        key = tf.transpose(tf.stack(all_link_outputs[:-1], axis=0), perm=[1, 0, 2])  # [B,j-1, link_embedding_dim]
        query = tf.reshape(link_o, [batch_size, 1, link_embedding_dim])  # [B,1, link_embedding_dim]

        # 使用加性注意力机制
        score = tf.keras.activations.tanh(link_atten_w_1(query) + link_atten_w_2(key))  # [b, j-1, d]
        logits = link_atten_w_a(score)  # [B, j-1,1] 表示选择该边作为输入的概率

        # 根据概率采样
        logits_reshape = tf.reshape(logits, [-1, 1])  # [B * (j-1),1]
        link = tf.random.categorical(tf.math.log(tf.concat([1. - logits_reshape, logits_reshape], axis=-1)),
                                     1)  # [B * (j-1),1]
        link = tf.reshape(link, [batch_size, j - 1])
        link_padding = tf.reshape(tf.pad(link, [[0, 0], [0, num_node - j + 1]]),
                                  [batch_size,  num_node])  # [B,  num_node]
        all_links.append(tf.cast(link_padding, dtype=tf.float32))

        inputs = link_embedding_layer(tf.reshape(link_padding,[batch_size,1,num_node]) ) # [B, 1, link_embedding_dim]

        # 计算loss，如果使用这个link_ce_loss计算梯度的话，lstm会趋向生成和采样结果相同的分布
        link_ce_loss = tf.losses.sparse_categorical_crossentropy(link,
                                                                 tf.concat([1. - logits, logits], axis=-1))  # [B, j-1]
        all_link_ce_loss.append(tf.reduce_sum(link_ce_loss, axis=-1))  # [B]

    all_link = tf.broadcast_to(tf.stack(all_links, axis=1),
                               shape=[tf.shape(input_tensor)[0], num_node, num_node])  # [B, num_node, num_node]
    link_ce_loss = tf.broadcast_to(tf.reduce_sum(tf.stack(all_link_ce_loss), axis=0),
                                   shape=[tf.shape(input_tensor)[0]])  # [B]

    model = tf.keras.Model(inputs=[input_tensor],
                           outputs=[all_link, link_ce_loss])
    return model
