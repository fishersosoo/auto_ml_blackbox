# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.state_ops import assign_sub

from auto_ml.tests.simple_text_classification import encode_input

from auto_ml.utils.tokenizer import PretrainTokenizer

from auto_ml.utils.data import TNewsData

from auto_ml.utils.controller import build_controller_model
from auto_ml.utils.dag import build_DAG_model


def fix_link(num_node):
    eye = tf.eye(num_node, dtype=tf.int32)[:num_node - 1]
    links = tf.concat([tf.zeros([1, num_node], dtype=tf.int32), eye], axis=0)
    return tf.expand_dims(links, axis=0)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.all_prob = []
        self.all_ce_loss = []
        self.all_link = []

    def on_epoch_end(self, epoch, logs=None):
        all_link, all_ce, all_prob = self.model.predict({"input": tf.ones([1, 1])})
        print(all_prob[0])
        self.all_prob.append(all_link)
        self.all_ce_loss.append(all_ce)
        self.all_prob.append(all_prob)


class DAGModel(tf.keras.Model):
    def __init__(self,
                 max_len,
                 input_dim,
                 num_node,
                 model_hidden_size,
                 num_class,
                 link_embedding_dim,
                 link_hidden_size,
                 M,
                 reward_const,
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
        self.controller = build_controller_model(num_node, link_hidden_size, controller_temperature=5.0,
                                                 controller_tanh_constant=2.5)
        self.controller_entropy_weight = 1e-5
        self.num_class = num_class
        self.num_node = num_node
        self.M = M
        self.reward_const = reward_const

        self.controller_baseline_dec = 0.999
        self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            5.0,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00035)
        self.g_loss = tf.losses.CategoricalCrossentropy(from_logits=True)
        self.c_loss = tf.losses.CategoricalCrossentropy(from_logits=True)

        self.model_acc = tf.keras.metrics.CategoricalAccuracy(name="model_acc")
        self.model_loss_tracker = tf.keras.metrics.Mean(name="model_loss")
        self.controller_reward_tracker = tf.keras.metrics.Mean(name="reward")
        self.controller_loss_tracker = tf.keras.metrics.Mean(name="controller_loss")
        self.controller_entropy_tracker = tf.keras.metrics.Mean(name="controller_entropy")
        self.controller_baseline_tracker = tf.keras.metrics.Mean(name="baseline")

    def train_step(self, data):
        # x, y = data
        input_embeddings_train = data["train_x"]["input_embeddings"]
        input_mask_train = data["train_x"]["input_mask"]
        label_train = data["train_y"]["label"]
        batch_size = tf.shape(data["train_x"]["input_embeddings"])[0]

        input_embeddings_eval = data["eval_x"]["input_embeddings"]
        input_mask_eval = data["eval_x"]["input_mask"]
        label_eval = data["eval_y"]["label"]
        # 生成M个子模型训练DAG
        all_sample_losses = None
        for i in range(self.M):
            all_link, _, _ = self.controller({"input": tf.ones([1, 1])})
            all_link = tf.broadcast_to(all_link, [batch_size, self.num_node, self.num_node])
            with tf.GradientTape() as tape:
                prob, label = self.model({"input_embeddings": input_embeddings_train,
                                          "input_mask": input_mask_train,
                                          "edges": all_link
                                          })
                model_loss = self.g_loss(tf.one_hot(label_train, self.num_class), prob)
                if all_sample_losses is None:
                    all_sample_losses = model_loss / self.M
                else:
                    all_sample_losses += model_loss / self.M
        grads = tape.gradient(all_sample_losses, self.model.trainable_weights)
        # grads = [tf.clip_by_norm(g, 0.25)
        #          for g in grads]
        self.g_optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.model_loss_tracker.update_state(all_sample_losses)
        self.model_acc.update_state(label_train, label)

        # 生成子模型训练控制器
        with tf.GradientTape() as tape:
            all_link, all_ce, _ = self.controller({"input": tf.zeros([1, 1])})
            all_link = tf.broadcast_to(all_link, [batch_size, self.num_node, self.num_node])
            all_ce = tf.broadcast_to(all_ce, [batch_size, self.num_node - 1])
            prob, label = self.model({"input_embeddings": input_embeddings_eval,
                                      "input_mask": input_mask_eval,
                                      "edges": all_link
                                      })
            child_model_loss = self.c_loss(tf.one_hot(label_eval, self.num_class), prob)  # [1]
            child_model_loss = tf.stop_gradient(child_model_loss)
            reward = self.reward_const / child_model_loss  # [1]

            entropy = tf.stop_gradient(all_ce * tf.exp(-all_ce))
            self.controller_entropy_tracker.update_state(entropy)

            reward += self.controller_entropy_weight * entropy
            self.controller_reward_tracker.update_state(reward)

            # baseline
            # pure_reward = reward
            # self.baseline = (1 - self.controller_baseline_dec) * (self.baseline - reward)
            baseline_update_op = assign_sub(self.baseline,
                ((1 - self.controller_baseline_dec) *
                 (self.baseline - tf.reduce_mean(reward))))
            with tf.control_dependencies([baseline_update_op]):
                reward = tf.identity(reward)
            self.controller_baseline_tracker.update_state(self.baseline)
            # link_label=tf.stack([1-tf.cast(all_link[:, 1:, :],tf.int32),tf.cast(all_link[:, 1:, :],tf.int32)],axis=-1)
            # link_ce_loss = tf.losses.binary_crossentropy(tf.stop_gradient(link_label), all_probs)  # [b,n-1,n]

            controller_loss = (all_ce) * (reward - self.baseline)
            self.controller_loss_tracker.update_state(controller_loss)

        grads = tape.gradient(controller_loss, self.controller.trainable_weights)
        self.c_optimizer.apply_gradients(zip(grads, self.controller.trainable_weights))
        return {"model_loss": self.model_loss_tracker.result(),
                "model_acc": self.model_acc.result(),
                "controller_entropy": self.controller_entropy_tracker.result(),
                "controller_loss": self.controller_loss_tracker.result(),
                "baseline": self.controller_baseline_tracker.result(),
                "reward": self.controller_reward_tracker.result()}

    def predict(self, x, **kwargs):
        return self.controller(x)

    @property
    def metrics(self):
        return [self.model_loss_tracker, self.model_acc]


if __name__ == '__main__':
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    MAX_LEN = 32
    num_class = 15
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    logdir = '/home/auto_ml/models/simple_dag'

    data = TNewsData("/home/auto_ml/data/source/TNewsData")
    tokenizer = PretrainTokenizer("/home/auto_ml/models/sgns.renmin.char")
    input_dim = tokenizer.dim

    enas_model = DAGModel(max_len=MAX_LEN,
                          input_dim=input_dim,
                          num_node=10,
                          model_hidden_size=64,
                          num_class=num_class,
                          link_embedding_dim=100,
                          link_hidden_size=100,
                          M=5,
                          reward_const=30.0)

    head = None

    training_x = encode_input(data["train"][:head], tokenizer, max_len=MAX_LEN)
    training_y = {
        "label": tf.convert_to_tensor(data["train"][:head]["label_id"], dtype=tf.int32),
        "prob": tf.one_hot(tf.convert_to_tensor(data["train"][:head]["label_id"], dtype=tf.int32), depth=num_class)
    }
    # eval_x = encode_input(data["eval"].sample()[:head], tokenizer, max_len=MAX_LEN)
    # eval_y = {
    #     "label": tf.convert_to_tensor(data["eval"][:head]["label_id"], dtype=tf.int32),
    #     "prob": tf.one_hot(tf.convert_to_tensor(data["eval"][:head]["label_id"], dtype=tf.int32), depth=num_class)
    # }
    data = tf.data.Dataset.from_tensor_slices(
        {"train_x": training_x, "eval_x": training_x, "train_y": training_y, "eval_y": training_y}).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    p_callback = CustomCallback()
    enas_model.compile()

    enas_model.fit(data,
                   epochs=20,
                   callbacks=[tensorboard_callback, p_callback]
                   )
    all_link, all_ce, all_prob = enas_model.controller({"input": tf.zeros([1, 1])})

# print(tf.math.softmax(all_probs)[:,:,:,1])
