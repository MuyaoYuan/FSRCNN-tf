import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class ESCPN(keras.Model):
    def __init__(self, n_colors, scale):
        super(ESCPN, self).__init__()
        self.n_colors = n_colors
        self.scale = scale
        self.Conv1 = keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu')
        self.Conv2 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.Conv3 = keras.layers.Conv2D(filters=n_colors*scale*scale, kernel_size=3, padding='same', activation='relu')
        self.Subpixel_layer = tf.nn.depth_to_space
        self.Conv4 = keras.layers.Conv2D(filters=n_colors, kernel_size=1, padding='same', activation=None)

    def call(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Subpixel_layer(x, self.scale)
        x = self.Conv4(x)
        return x

if __name__ == '__main__':
    net = ESCPN(3, 2)
    # net.build((1, 240, 320, 3))
    # print(net.summary())
    # net.save('trained_model/ESCPN')


    input_dummy = tf.ones([1, 240, 320, 3 ])
    output_dummy = net(input_dummy)
    print(output_dummy)
    net.save('trained_model/ESCPN')

    # tfjs_target_dir = 'tfjs_model'
    # tfjs.converters.save_keras_model(net, tfjs_target_dir)