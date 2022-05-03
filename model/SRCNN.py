import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
# import tensorflowjs as tfjs
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SRCNN(keras.Model):
    def __init__(self, n_colors):
        super(SRCNN, self).__init__()
        self.n_colors = n_colors
        self.Conv1 = keras.layers.Conv2D(filters=64, kernel_size=9, padding='same', activation='relu')
        self.Conv2 = keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation='relu')
        self.Conv3 = keras.layers.Conv2D(filters=n_colors, kernel_size=5, padding='same', activation=None)

    def call(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        return x

def bulid_model(n_colors):
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=9, padding='same', activation='relu'),
        keras.layers.Conv2D(filters=32, kernel_size=1, padding='same', activation='relu'),
        keras.layers.Conv2D(filters=n_colors, kernel_size=5, padding='same', activation=None)
        ])
    return model


if __name__ == '__main__':
    net = bulid_model(3)
    # net.build((1, 480, 640, 3))
    # print(net.summary())

    # input_dummy = tf.ones([1, 480, 640, 3])
    # output_dummy = net(input_dummy)
    # print(output_dummy)

    # tfjs_target_dir = 'tfjs_model'
    # tfjs.converters.save_keras_model(net, tfjs_target_dir)

    # 性能测试
    input_dummy = tf.ones([1, 480, 640, 3])
    for _ in range(100000):
        output_dummy = net(input_dummy)
