import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data import DIV2K

class Reloader:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path, compile=False)
        self.dataset_valid = DIV2K(subset='valid').dataset()

    def lossShow(train_loss_arr_path, valid_loss_arr_path, save_path):
        train_loss_arr = np.load(train_loss_arr_path)
        valid_loss_arr = np.load(valid_loss_arr_path)
        epochs = len(train_loss_arr)
        epochs_arr = np.arange(epochs) + 1
        plt.figure()
        plt.plot(epochs_arr, train_loss_arr, 'b', label='train_loss')
        plt.plot(epochs_arr, valid_loss_arr, 'y', label='valid_loss')
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoches')
        plt.title('loss_curve')
        plt.savefig(save_path)


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
    reloader = Reloader(model_path = 'trained_model/FSRCNN/FSRCNN.h5')
    reloader.reload()
    Reloader.lossShow('trained_model/FSRCNN/train_loss_arr.npy', 'trained_model/FSRCNN/valid_loss_arr.npy', 'trained_model/FSRCNN/loss_curve_fsrcnn')

