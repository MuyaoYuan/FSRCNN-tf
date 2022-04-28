import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from data import DIV2K

class Reloader:
    def __init__(self):
        self.model = keras.models.load_model('trained_model/SRCNN.h5', compile=False)
        self.dataset_valid = DIV2K(subset='valid').dataset()

    def reload(self):
        dataIter = iter(self.dataset_valid)
        testItem = dataIter.next()
        output = self.model(testItem)
        inputItem = testItem[0][0].numpy()
        labelItem = testItem[1][0].numpy()
        outputItem = output[0].numpy()
        # print(outputItem.shape)
        input_img = Image.fromarray(np.uint8(inputItem))
        label_img = Image.fromarray(np.uint8(labelItem))
        output_img = Image.fromarray(np.uint8(outputItem))
        input_img.save('input.png')
        label_img.save('label.png')
        output_img.save('output.png')


if __name__  == '__main__':
    reloader = Reloader()
    reloader.reload()

