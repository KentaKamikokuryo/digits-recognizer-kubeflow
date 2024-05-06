from model.model import DigitsRecognizerCNN
import tensorflow as tf
import numpy as np
import os
from kfp.dsl import InputPath, OutputPath

def train_model(train_data_dir: str, test_data_dir: str, model_dir: str, params: dict):

    # データの読み込み
    train_data = tf.data.experimental.load(train_data_dir)
    test_data = tf.data.experimental.load(test_data_dir)

    # モデルの作成
    model = DigitsRecognizerCNN(params)

    # モデルのコンパイル
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"))

    # モデルの訓練
    model.fit(train_data, validation_data=test_data, epochs=params['epochs'], batch_size=params['batch_size'])

    # モデル保存用のディレクトリを作成
    os.makedirs(model_dir, exist_ok=True)

    # モデルの保存
    model.save(f"{model_dir}/model.h5")
