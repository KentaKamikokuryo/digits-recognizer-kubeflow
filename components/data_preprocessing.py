import tensorflow as tf


def load_and_preprocess_data():
    """
    Loads and preprocesses the MNIST dataset.

    Returns:
        Tuple: A tuple containing the preprocessed training and testing data.
            The training data is a tuple (x_train, y_train) where x_train is the
            normalized and reshaped input images and y_train is the one-hot encoded
            labels. The testing data is a tuple (x_test, y_test) with the same format.
    """
    # MNISTデータセットを読み込む
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # データを正規化する（0-255の値を0-1にスケーリング）
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # データの形状を変更する（モデル入力用に適合させる）
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # ラベルをone-hotエンコーディングする
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def save_data(x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test: tf.Tensor, save_dir: str):
    """
    Saves the preprocessed data to disk.

    Args:
        x_train (tf.Tensor): The preprocessed training input images.
        y_train (tf.Tensor): The one-hot encoded training labels.
        x_test (tf.Tensor): The preprocessed testing input images.
        y_test (tf.Tensor): The one-hot encoded testing labels.
        save_dir (str): The directory where the data will be saved.
    """
    
    # 訓練データセットの保存
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    tf.data.experimental.save(train_data, save_dir + '/train')

    # テストデータセットの保存
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    tf.data.experimental.save(test_data, save_dir + '/test')

def load_data(data_dir: str):
    """
    Loads the preprocessed data from disk.

    Args:
        data_dir (str): The directory where the data is saved.

    Returns:
        Tuple: A tuple containing the preprocessed training and testing data.
            The training data is a tuple (x_train, y_train) where x_train is the
            normalized and reshaped input images and y_train is the one-hot encoded
            labels. The testing data is a tuple (x_test, y_test) with the same format.
    """

    # 保存されたデータセットの読み込み
    train_data = tf.data.experimental.load(
        data_dir + '/train', 
        element_spec=(tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float64), 
                      tf.TensorSpec(shape=(10,), dtype=tf.float64)))
    
    test_data = tf.data.experimental.load(
        data_dir + '/test', 
        element_spec=(tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float64), 
                      tf.TensorSpec(shape=(10,), dtype=tf.float64)))

    return train_data, test_data
