import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

from model.SRCNN import bulid_model as bulid_SRCNN
from model.FSRCNN import bulid_model as build_FSRCNN

class Converter:
    def __init__(self, model_path, tfjs_target_dir):
        self.trained_model = keras.models.load_model(model_path, compile=False)
        self.save_model = build_FSRCNN(scale_factor=2, num_channels=3)
        self.save_model.build([1, 240, 320, 3])
        for layer in self.save_model.layers:
            try:
                layer.set_weights(self.trained_model.get_layer(name=layer.name).get_weights())
            except:
                print("Could not transfer weights for layer {}".format(layer.name))
        print(self.save_model.summary())
        self.tfjs_target_dir = tfjs_target_dir


    def convert(self):
        tfjs.converters.save_keras_model(self.save_model, self.tfjs_target_dir)


if __name__  == '__main__':
    converter = Converter('trained_model/FSRCNN/FSRCNN.h5', 'tfjs_model/FSRCNN')
    converter.convert()