import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def bulid_model(scale_factor, num_channels=3, d=56, s=12, m=4):

    
    model = keras.models.Sequential([
        # fisrt part
        keras.layers.Conv2D(filters=d, kernel_size=5, padding='same', activation=None),
        keras.layers.PReLU(shared_axes=[1, 2]), # share parameters across space
        # mid part - 1
        keras.layers.Conv2D(filters=s, kernel_size=1, padding='same', activation=None),
        keras.layers.PReLU(shared_axes=[1, 2])
        ])

    # mid part - 2
    for _ in range(m):
        model.add(keras.layers.Conv2D(filters=s, kernel_size=3, padding='same', activation=None))
        model.add(keras.layers.PReLU(shared_axes=[1, 2]))

    # mid part - 3
    model.add(keras.layers.Conv2D(filters=d, kernel_size=1, padding='same', activation=None))
    model.add(keras.layers.PReLU(shared_axes=[1, 2]))

    # last part
    model.add(keras.layers.Conv2DTranspose(filters=num_channels, kernel_size=9, strides=scale_factor, padding='same', output_padding=scale_factor-1, activation=None))

    # 初始化
    # TODO
    
    return model


if __name__ == '__main__':
    net = bulid_model(scale_factor = 2)
    net.build((1, 240, 320, 3))
    print(net.summary())

    # input_dummy = tf.ones([1, 480, 640, 3])
    # output_dummy = net(input_dummy)
    # print(output_dummy)

    # tfjs_target_dir = 'tfjs_model/FSRCNN'
    # os.makedirs(tfjs_target_dir, exist_ok=True)
    # tfjs.converters.save_keras_model(net, tfjs_target_dir)

    # # 性能测试
    # input_dummy = tf.ones([1, 480, 640, 3])
    # for _ in range(100000):
    #     output_dummy = net(input_dummy)
