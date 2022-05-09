import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

from model.SRCNN import bulid_model

class Converter:
    def __init__(self):
        self.trained_model = keras.models.load_model('trained_model/SRCNN.h5', compile=False)
        self.save_model = bulid_model(n_colors=3)
        self.save_model.build([1, 240, 320, 3 ])
        for layer in self.save_model.layers:
            try:
                layer.set_weights(self.trained_model.get_layer(name=layer.name).get_weights())
            except:
                print("Could not transfer weights for layer {}".format(layer.name))
        print(self.save_model.summary())
        self.tfjs_target_dir = 'tfjs_model'


    def convert(self):
        tfjs.converters.save_keras_model(self.save_model, self.tfjs_target_dir)


if __name__  == '__main__':
    converter = Converter()
    converter.convert()