# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import standard_ops

MAX_LEN = 32  # 最大文本长度
INPUT_DIM = 300  # 输入维度
DIM = 128  # 隐含层维度


def build_DAG_model(num_node, max_len, input_dim, num_class, hidden_size):
    """

    Args:
        num_node: 节点数量
        max_len:
        input_dim:
        num_class:
        hidden_size:

    Returns:

    """
    input_embeddings = tf.keras.Input([max_len, input_dim], name="input_embeddings")
    input_mask = tf.keras.Input([max_len], name="input_mask", dtype=tf.bool)
    edges = tf.keras.Input([num_node, num_node], name="edges")  # 对于整个batch，edges都是一样的
    link=edges[0] # [num_node, num_node]
    nodes = [tf.keras.layers.LSTM(hidden_size, return_sequences=True) for one in range(num_node)]

    node_output = []
    for j, node in enumerate(nodes):
        if j == 0:
            y = nodes[j](input_embeddings)
            node_output.append(y)
        else:
            weight=tf.expand_dims(link[j,:j],axis=0) # [1, j]
            all_node_output = tf.stack(node_output, axis=-1)  # [b,l,d,j]
            node_input=tf.squeeze(tf.matmul(all_node_output,weight,transpose_b=True),axis=[-1])  # [b,l,d]
            # input_to_j = tf.reshape(edges[:, j, :j], [-1, 1, 1,j])  # [b 1,1,j]
            # node_input = tf.reduce_sum(input_to_j * all_node_output, axis=-1)  # [b,l,d]
            y = nodes[j](node_input)
            node_output.append(y)
    out_degree = tf.reduce_sum(link, axis=0)  # 各节点出度 [n]
    has_out=tf.cast(out_degree,dtype=tf.bool)
    is_leaf=tf.cast(tf.math.logical_not(has_out),tf.float32)
    is_leaf = tf.expand_dims(is_leaf,axis=-1)  #  [n,1]
    sum_node_input = tf.stack(node_output, axis=-1)  # [b,l,d,n]
    sum_node_input=tf.squeeze(tf.matmul(sum_node_input,is_leaf),axis=-1) # [b,l,d]
    # sum_node_input = tf.reduce_sum(is_leaf * sum_node_input, axis=1)  # [b,l,d]
    # prob = tf.keras.layers.Dense(num_class, activation='relu')(sum_node_input[:, 0, :])  # [B,class_num]
    # label = tf.keras.layers.Lambda(tf.argmax, arguments={'axis': -1}, name="label")(prob)
    y=tf.keras.layers.LSTM(hidden_size)(sum_node_input)
    prob = tf.keras.layers.Dense(num_class, activation='relu')(y)  # [B,class_num]
    label = tf.keras.layers.Lambda(tf.argmax, arguments={'axis': -1}, name="label")(prob)

    model = tf.keras.Model(inputs=[input_embeddings,  input_mask,edges], outputs=[prob, label])
    return model


class maskInitializer(tf.keras.initializers.Initializer):
    def __init__(self, node_num):
        self.node_num = node_num

    def __call__(self, shape, dtype=None):
        edges = []
        for i in range(self.node_num):
            # node-i output to where
            edges.append([0] * (i + 1) + [1] * (self.node_num - i - 1))
        return tf.Variable(np.array(edges).astype(float), dtype=tf.float32)


@tf.function
def stack_func(tensor, all_edges, node_num, num_layers):
    batch_size = tf.shape(tensor)[0]
    all_edges = tf.stack(all_edges)
    all_edges = tf.expand_dims(all_edges, axis=0)
    return tf.broadcast_to(all_edges, shape=[batch_size, num_layers, node_num * node_num])


if __name__ == '__main__':
    dag = build_DAG_model(5, 32, 300, 15, 128)
