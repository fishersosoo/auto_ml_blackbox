# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import standard_ops

MAX_LEN = 32  # 最大文本长度
INPUT_DIM = 300  # 输入维度
DIM = 128  # 隐含层维度


def build_DAG_model(num_node, max_len, input_dim, num_class, hidden_size):
    input_embeddings = tf.keras.Input([max_len, input_dim], name="input_embeddings")
    input_mask = tf.keras.Input([max_len], name="input_mask", dtype=tf.bool)
    edges = tf.keras.Input([num_node, num_node], name="edges")  # 对于整个batch，edges都是一样的
    edges = tf.stop_gradient(edges)
    nodes = [tf.keras.layers.GRU(hidden_size, return_sequences=True) for one in range(num_node)]
    node_output = []
    for j, node in enumerate(nodes):
        if j == 0:
            y = nodes[j](input_embeddings)
            node_output.append(y)
        else:
            all_node_output = tf.stack(node_output, axis=2)  # [b,l,j-1,d]
            input_to_j = tf.reshape(edges[:, :j, j], [-1, 1, j - 1, 1])  # [b,1 j-1,1]
            node_input = tf.reduce_sum(input_to_j * all_node_output, axis=2)  # [b,l,d]
            y = nodes[j](node_input)
            node_output.append(y)
    is_leaf = tf.reduce_sum(edges, axis=2)  # [b, n]
    is_leaf = tf.reshape(tf.keras.activations.relu(1 - is_leaf), [-1, 1, node_num, 1])  # [B, 1, N, 1]
    sum_node_input=tf.stack(node_output, axis=2)  # [b,l,n,d]
    sum_node_input=tf.reduce_sum(is_leaf*sum_node_input, axis=2) # [b,l,d]
    output_y = tf.keras.layers.Dense(num_class,activation='sigmoid')(sum_node_input[:, 0, :])  # [B,class_num]
    prob = tf.keras.layers.Softmax(name="prob")(output_y)
    label = tf.keras.layers.Lambda(tf.argmax, arguments={'axis': -1}, name="label")(output_y)
    model = tf.keras.Model(inputs=[input_embeddings, edges,input_mask], outputs=[prob, label])
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







class DAGModel(tf.keras.Model):
    def __init__(self, model, controller, class_num, node_num):
        """

        Args:
            model: DAG模型
            controller: controller模型
            class_num: 类别数量
            node_num: 节点数量
        """
        super(DAGModel, self).__init__()
        self.model = model
        self.controller = controller
        self.class_num = class_num
        self.node_num = node_num
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.g_loss = tf.losses.categorical_crossentropy

    def train_step(self, data):
        """
        TODO:还有问题，清对照《算法流程.docx》修改
        自定义训练循环
        Args:
            data: 训练输入

        Returns:

        """
        input_embeddings = data[0]  # [b, MAX_LEN, INPUT_DIM]
        labels = data[1]  # [b]
        batch_size = tf.shape(input_embeddings)[0]

        all_edges = self.controller(tf.zeros([1, 1]))  # [1, controller_num_layers,node_num * node_num]
        all_edges = tf.squeeze(all_edges, axis=[0])  # [controller_num_layers,node_num * node_num]
        # M = tf.shape(all_edges)[0]  # controller_num_layers
        M = 5
        all_edges = tf.reshape(all_edges, [M, self.node_num, self.node_num])
        all_grads = []
        all_model_loss = []
        for i in range(M):
            with tf.GradientTape() as tape:
                edges = tf.reshape(all_edges[M], [1, -1])
                edges = tf.tile(edges, [BATCH_SIZE, 1])
                predictions = self.model(inputs=[input_embeddings, edges])
                model_loss = self.g_loss(tf.one_hot(labels, depth=self.class_num), predictions)
            grads = tape.gradient(model_loss, self.model.trainable_weights)
            all_grads.append(grads)
            all_model_loss.append(model_loss)
        self.g_optimizer.apply_gradients(
            zip(tf.reduce_mean(tf.stack(all_grads), axis=0), self.model.trainable_weights)
        )

        all_edges = self.controller(tf.zeros([1, 1]))  # [1, controller_num_layers,node_num * node_num]
        all_edges = tf.squeeze(all_edges, axis=[0])  # [controller_num_layers,node_num * node_num]
        all_c_grads = []
        all_c_loss = []
        for i in range(M):
            with tf.GradientTape() as tape:
                edges = tf.reshape(all_edges[M], [1, -1])
                edges = tf.tile(edges, [BATCH_SIZE, 1])
                predictions = self.model(inputs=[input_embeddings, edges])
                controller_loss = self.g_loss(tf.one_hot(labels, depth=self.class_num), predictions)

        with tf.GradientTape() as tape:
            repeat_embeddings = tf.tile(input_embeddings, [M, 1, 1])  # [B*M,node_num * node_num]
            repeat_labels = tf.tile(labels, [M])
            repeat_edges = tf.tile(all_edges, [batch_size, 1, 1])
            predictions = self.model(inputs=[repeat_embeddings, repeat_edges])
            controller_loss = self.g_loss(tf.one_hot(repeat_labels, depth=self.class_num), predictions)
        grads = tape.gradient(controller_loss, self.controller.trainable_weights)
        print(grads)
        self.c_optimizer.apply_gradients(
            zip(grads, self.controller.trainable_weights)
        )
        return {'model_loss': tf.reduce_mean(tf.stack(all_model_loss), axis=0), "controller_loss": controller_loss}


if __name__ == '__main__':
    # 使用随机数据作为训练数据
    # TODO:后面会改为使用公开数据集数据
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (np.random.random([16 * 10, MAX_LEN, INPUT_DIM]), np.random.random_integers(0, 1, [16 * 10])))
    BATCH_SIZE = 16
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    node_num = 6
    class_num = 2
    M = 4  # 每次训练蒙特卡洛采样次数
    dag = DAGBuilder(node_num)
    model = dag.buildDAG(class_num)
    # print(model.trainable_weights)
    controller = dag.buildController(M)
    enas = DAGModel(model=model, controller=controller, class_num=2, node_num=6)
    enas.compile()
    enas.fit(train_dataset, epochs=10)

    print(tf.reshape(enas.controller(tf.zeros([1, 1])), [1, M, node_num, node_num]))
