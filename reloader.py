import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data import DIV2K

class Reloader:
    def __init__(self):
        self.model = keras.models.load_model('trained_model/SRCNN.h5', compile=False)
        self.dataset_valid = DIV2K(subset='valid').dataset()

    def lossShow():
        train_loss_arr = np.load('trained_model/train_loss_arr.npy')
        valid_loss_arr = np.load('trained_model/valid_loss_arr.npy')
        epochs = len(train_loss_arr)
        epochs_arr = np.arange(epochs) + 1
        plt.figure()
        plt.plot(epochs_arr, train_loss_arr, 'b', label='train_loss')
        plt.plot(epochs_arr, valid_loss_arr, 'y', label='valid_loss')
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoches')
        plt.title('loss_curve')
        plt.savefig('trained_model/loss_curve.png')


    def reload(self):
        dataIter = iter(self.dataset_valid)
        testItem = dataIter.next()
        output = self.model(testItem[0])
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
    Reloader.lossShow()

