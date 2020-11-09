# coding=utf-8
"""
ENAS算法控制器模型
"""
import tensorflow as tf
import numpy as np


def build_controller_model(num_node, hidden_size, controller_temperature, controller_tanh_constant):
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

    input_tensor = tf.keras.Input(shape=[1], name="input_tensor", dtype=tf.float32)  # [B,1]
    batch_size = tf.shape(input_tensor)[0]

    link_embedding_layer = tf.keras.layers.Embedding(input_dim=num_node - 1, output_dim=hidden_size,
                                                     name="link_embedding_layer")

    link_lstm_layer = tf.keras.layers.LSTM(hidden_size, return_sequences=False, return_state=True, trainable=True,
                                           recurrent_activation=None, name="link_lstm")
    init_link_input = tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None, trainable=True,
                                            name="init_link_inputs")
    # 加性注意力层
    link_atten_w_1 = tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None, trainable=True, name="w_1")
    link_atten_w_2 = tf.keras.layers.Dense(hidden_size, use_bias=False, activation=None, trainable=True, name="w_2")
    link_atten_w_a = tf.keras.layers.Dense(1, use_bias=False, activation=None, trainable=True, name="w_a")

    # 初始化输入
    init_link_embedding = init_link_input(input_tensor)  # [B, link_embedding_dim]
    all_h = [tf.broadcast_to(tf.zeros(shape=[1, hidden_size]),
                             shape=[batch_size, hidden_size])]  # 连接向量lstm层的输出 [j, B, link_embedding_dim]
    all_h_w = [tf.broadcast_to(tf.zeros(shape=[1, hidden_size]),
                               shape=[batch_size, hidden_size])]
    all_links = [tf.broadcast_to(tf.zeros(shape=[1, num_node]),
                                 shape=[batch_size, num_node])]  # 生成的连接向量 [b,1,n],最后会堆叠成[b,n,n]

    all_ce_loss = []  # 损失[B,num_node-1]
    all_prob = []  # [B,num_node-1(stack axis), num_node]

    lstm_input = tf.expand_dims(init_link_embedding, 1)  # [B,1, link_embedding_dim]
    lstm_state = None

    for j in range(2, num_node + 1):
        _, link_c, link_h = link_lstm_layer(lstm_input,
                                            initial_state=lstm_state)  # [B, link_embedding_dim]
        lstm_state = [link_c, link_h]
        all_h.append(link_h)  # [j, B, link_embedding_dim]

        all_h_w.append(link_atten_w_1(link_h))
        query = link_atten_w_2(link_h)

        key = tf.transpose(tf.stack(all_h_w[:-1], axis=0), perm=[1, 0, 2])  # [B,j-1, link_embedding_dim]
        query = tf.reshape(query, [batch_size, 1, hidden_size])  # [B,1, link_embedding_dim]
        query = tf.nn.tanh(query + key)  # [B,j-1, link_embedding_dim]

        logits = link_atten_w_a(query)  # [B,j-1, 1]
        logits = logits / controller_temperature
        logits = controller_tanh_constant * tf.nn.tanh(logits)
        logits = tf.squeeze(logits, -1)  # [B, j-1] 前置节点概率

        prob = tf.pad(logits, [[0, 0], [0, num_node - j + 1]])  # [B, num_node]
        all_prob.append(prob)

        # 根据概率采样获得前置节点id和前置节点向量表示
        input_node_id = tf.squeeze(tf.random.categorical(logits, 1), axis=[-1])  # [B]
        link = tf.one_hot(input_node_id, depth=num_node)  # [B,num_node]
        link_embedding = link_embedding_layer(tf.expand_dims(input_node_id, -1))  # [B,1,link_embedding_dim]

        # 计算损失
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=tf.stop_gradient(input_node_id),
                                                                 name=f"controller_ce_{j}")  # [B]
        all_links.append(link)
        all_ce_loss.append(ce_loss)

        lstm_input = link_embedding  # [B, 1, link_embedding_dim]
    all_prob = tf.stack(all_prob, 1)  # [B, num_node-1, num_node]
    all_links = tf.stack(all_links, 1)
    all_ce_loss = tf.stack(all_ce_loss, axis=-1)  # [B,num_node-1]
    model = tf.keras.Model(inputs=[input_tensor],
                           outputs=[all_links, all_ce_loss, all_prob])
    return model
