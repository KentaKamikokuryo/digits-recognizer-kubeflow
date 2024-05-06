import tensorflow as tf

class DigitsRecognizerCNN(tf.keras.Model):
    def __init__(self, params):
        super(DigitsRecognizerCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(params['conv1_filters'], (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(params['conv2_filters'], (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(params['dropout_rate'])
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)
