import numpy as np
import tensorflow as tf
from typing import Tuple
from sklearn.metrics import confusion_matrix
import pandas as pd

def prediction(model_dir: str, test_data: Tuple[tf.Tensor, tf.Tensor], mlpipeline_ui_metadata_dir: str) -> int:
    """
    Predicts the digit in the input image.

    Args:
        model_dir (str): The directory where the model is saved.
        image (np.ndarray): The input image.

    Returns:
        int: The predicted digit.
    """
    # データの読み込み
    X_test, y_test = test_data

    # モデルの読み込み
    model = tf.keras.models.load_model(f"{model_dir}/model.h5")

    # 画像の形状を変更

    # 予測
    prediction = model.predict(X_test)

    y_pred = np.argmax(prediction, axis=1)

    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    vocab = list(np.unique(y_test))

    # confusion matrix ペアのリスト
    cm_pairs = []
    for target_index, target_row in enumerate(cm):
        for pred_index, count in enumerate(target_row):
            cm_pairs.append((vocab[target_index], vocab[pred_index], count))

    # pandas DataFrame に変換
    df = pd.DataFrame(cm_pairs, columns=['target', 'pred', 'count'])

    # 'target' と 'pred' をストリングに変換
    df['target'] = df['target'].astype(str)
    df['pred'] = df['pred'].astype(str)

    # Kubeflow metric metadata を作成
    metadata = {
        "outputs": [
            {
                "type": "confusion_matrix",
                "format": "csv",
                "schema": [
                    {
                        "name": "target",
                        "type": "CATEGORY"
                    },
                    {
                        "name": "pred",
                        "type": "CATEGORY"
                    },
                    {
                        "name": "count",
                        "type": "NUMBER"
                    }
                ],
                "source": df.to_csv(header=False, index=False),
                "storage": "inline",
                "labels": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ]
            }
        ]
    }

    with open(mlpipeline_ui_metadata_dir, "w") as metadata_file:
        import json
        json.dump(metadata, metadata_file)

    conf

