# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import standard_ops

MAX_LEN = 32  # 最大文本长度
INPUT_DIM = 300  # 输入维度
DIM = 128  # 隐含层维度


def weihgt(w, x):
    # w[a,b] * x[..., a] →[... , b]
    rank = x.shape.rank
    outputs = standard_ops.tensordot(x, w, [[rank - 1], [0]])
    return outputs


def random_connection(node_num):
    edges = []
    for i in range(node_num):
        # node-i output to where
        edges.append([0] * (i + 1) + np.random.random_integers(1, 1, [node_num - i - 1]).tolist())
    return np.array(edges)


class EdgesMaskLayer(tf.keras.layers.Layer):
    def __init__(self, node_num):
        super(EdgesMaskLayer, self).__init__()
        self.node_num = node_num
        self.mask = self.add_weight(shape=[node_num, node_num], trainable=False, initializer=maskInitializer(node_num))

    def call(self, inputs, **kwargs):
        mask = tf.reshape(self.mask, [1, 1, self.node_num * self.node_num])
        return inputs * mask


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


class DAGBuilder():
    def __init__(self, node_num):
        self.node_num = node_num

    def buildController(self, controller_num_layers):
        num_layers = controller_num_layers
        node_num = self.node_num
        mask_layer = EdgesMaskLayer(node_num=node_num)

        input_tensor = tf.keras.Input(shape=[1])

        inputs = tf.zeros([1, 1, node_num * node_num])
        lstm = tf.keras.layers.LSTM(node_num * node_num, return_sequences=True, return_state=True,
                                    activation=tf.keras.activations.sigmoid, bias_initializer='glorot_uniform')
        state = None
        all_edges = []
        all_prob = []
        for layer_id in range(1, num_layers + 1):
            outputs, final_memory_state, final_carry_state = lstm(inputs, initial_state=state)
            state = [final_memory_state, final_carry_state]
            logits = mask_layer(outputs)  # [1,1, node_num*node_num]
            all_prob.append(logits)
            logits = tf.reshape(logits, [node_num * node_num])

            logits = tf.stack([1 - logits, logits])  # [2, node_num * node_num]
            edges = tf.cast(tf.random.categorical(tf.transpose(logits), 1), dtype=tf.float32)  # [node_num * node_num,1]
            edges = mask_layer(tf.reshape(edges, [1, 1, node_num * node_num]))
            edges = tf.squeeze(edges, axis=[0, 1])
            all_edges.append(edges)
            inputs = tf.reshape(edges, [1, 1, node_num * node_num])

        all_edges = tf.stack(all_edges)

        all_edges = tf.broadcast_to(all_edges, shape=[tf.shape(input_tensor)[0], num_layers, node_num * node_num])

        # all_edges=tf.keras.layers.Lambda(stack_func)(inputs)
        controller = tf.keras.Model(inputs=input_tensor,
                                    outputs=all_edges)  # [b,controller_num_layers,node_num * node_num]
        return controller

    def buildDAG(self, class_num):
        """
        TODO:还有问题，清对照《算法流程.docx》修改

        Args:
            class_num:

        Returns:

        """
        node_num = self.node_num
        input = tf.keras.Input(shape=[MAX_LEN, INPUT_DIM])
        edges = tf.keras.Input(shape=[node_num, node_num])
        nodes = [tf.keras.layers.GRU(DIM, return_sequences=True, activation=tf.keras.activations.relu) for one in
                 range(node_num)]
        node_output = []
        for j, node in enumerate(nodes):
            if j == 0:
                y = nodes[j](input)
                node_output.append(y)
            else:
                mask = tf.reshape(edges[:, :j, j], [-1, j, 1, 1])  # [b,j,1]
                mask = tf.tile(mask, [1, 1, MAX_LEN, DIM])  # [b, j, L, D]
                stacked = tf.stack(node_output, axis=1)  # [b, j, L, D]
                stacked = tf.multiply(mask, stacked)  # [b, j, L, D]
                y = tf.reduce_sum(stacked, axis=1)  # [b, L, D]
                node_output.append(y)
        is_leaf = tf.reduce_sum(edges, axis=2)  # [b, n]
        is_leaf = tf.reshape(tf.keras.activations.relu(1 - is_leaf), [-1, 1, 1, node_num])  # [B, 1, 1, N]
        is_leaf = tf.tile(is_leaf, [1, MAX_LEN, DIM, 1])  # [B,L,D,N]
        sum_node = tf.transpose(tf.stack(node_output, axis=1), perm=[0, 2, 3, 1])  # [B,L,D,N]
        sum_node = tf.multiply(is_leaf, sum_node)
        sum_node = tf.reduce_sum(sum_node, axis=-1)  # [B,L,D]
        output_y = tf.keras.layers.Dense(class_num)(sum_node[:, 0, :])  # [B,class_num]
        model = tf.keras.Model(inputs=[input, edges], outputs=output_y)
        return model


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
        M = tf.shape(all_edges)[0]  # controller_num_layers
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
    enas.fit(train_dataset, epochs=100)
    # features = tf.Variable((np.random.random([1, 3])), dtype=tf.float32)
    # model(features)
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(),
    #               loss=tf.keras.losses.mean_squared_error,
    #               metrics=[tf.keras.metrics.mean_squared_error])
    # logdir = r'Z:\auto_ml\models'
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    # model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback])
    print(tf.reshape(enas.controller(tf.zeros([1, 1])), [1, M, node_num, node_num]))
