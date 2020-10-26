# encoding=utf-8
"""
使用TNew数据集进行文本分类的例子
"""
import tensorflow as tf
import numpy as np
from pandas import DataFrame
from auto_ml.utils.data import TNewsData
from auto_ml.utils.tokenizer import PretrainTokenizer


def encode_input(df, tokenizer, max_len):
    """
    文本转化为向量
    Args:
        df (DataFrame):
        tokenizer (PretrainTokenizer):
    """
    input_embeddings = []
    input_mask = []
    for sentence in df["sentence"]:
        embedding, mask = tokenizer.word2vec(sentence, max_len)
        input_mask.append(mask)
        input_embeddings.append(embedding)
    inputs = {
        "input_embeddings": tf.convert_to_tensor(np.array(input_embeddings)),
        "input_mask": tf.convert_to_tensor(np.array(input_mask))
    }
    return inputs


def build_model(max_len, input_dim, num_class):
    """
    构建一个LSTM、CNN、池化、全连接的分类模型
    Args:
        max_len:文本最大长度
        input_dim:文本向量维度
        num_class:类别数量

    Returns:

    """
    # 定义模型输入，会根据name从输入字典中查找对应列
    input_embeddings = tf.keras.Input([max_len, input_dim], name="input_embeddings")
    input_mask = tf.keras.Input([max_len], name="input_mask", dtype=tf.bool)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(input_embeddings, mask=input_mask)
    x = tf.transpose(tf.keras.layers.Conv1D(max_len, 3, padding='same')(x), [0, 2, 1])
    x = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dense(num_class)(x)
    prob = tf.keras.layers.Softmax(name="prob")(x)
    label = tf.keras.layers.Lambda(tf.argmax, arguments={'axis': -1}, name="label")(x)
    model = tf.keras.Model(inputs=[input_embeddings, input_mask], outputs=[prob, label])
    return model


if __name__ == '__main__':
    MAX_LEN = 32
    num_class = 15
    logdir = '/home/auto_ml/models/simple_text_classification'

    data = TNewsData("/home/auto_ml/data/source/TNewsData")
    tokenizer = PretrainTokenizer("/home/auto_ml/models/sgns.renmin.char")
    # 使用部分或者全部数据进行训练
    head = len(data["train"])
    training_x = encode_input(data["train"][:head], tokenizer, max_len=MAX_LEN)
    training_y = {
        "label": tf.convert_to_tensor(data["train"][:head]["label_id"], dtype=tf.int32),
        "prob": tf.one_hot(tf.convert_to_tensor(data["train"][:head]["label_id"], dtype=tf.int32), depth=num_class)
    }
    # 构建验证数据
    eval_head = len(data["eval"])
    eval_x = encode_input(data["eval"][:eval_head], tokenizer, max_len=MAX_LEN)
    eval_y = {
        "label": tf.convert_to_tensor(data["eval"][:eval_head]["label_id"], dtype=tf.int32),
        "prob": tf.one_hot(tf.convert_to_tensor(data["eval"][:eval_head]["label_id"], dtype=tf.int32), depth=num_class)
    }
    eval_dataset = tf.data.Dataset.from_tensor_slices((eval_x, eval_y)).batch(64, True)

    model = build_model(MAX_LEN, tokenizer.dim, num_class)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,update_freq="batch")
    # 构建模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss={"prob": tf.keras.losses.CategoricalCrossentropy()},
                  metrics={"label": tf.keras.metrics.CategoricalAccuracy(name="acc")},
                  )
    # 训练
    model.fit(x=training_x,
              y=training_y,
              batch_size=64,
              epochs=128,
              validation_data=eval_dataset,
              validation_steps=10,
              shuffle=True,
              callbacks=[tensorboard_callback]
              )
    model.save("/home/auto_ml/models/simple_text_classification")
