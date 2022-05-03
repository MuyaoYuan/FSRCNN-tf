import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 要放在 import tensorflow as tf 前面才会起作用 ！！！
import tensorflow as tf
from tensorflow import keras
import numpy as np

from model.SRCNN import bulid_model
from data import DIV2K

class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = bulid_model(n_colors=args.n_colors)
        self.dataset_train = DIV2K().dataset(repeat_count=1) # repeat_count=None 会一直重复
        self.dataset_valid = DIV2K(subset='valid').dataset(repeat_count=1)
        self.lossFun = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam()
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.test_loss = keras.metrics.Mean(name='test_loss')
        self.epochs = args.epochs

    @tf.function
    def train_step(self, item):
        with tf.GradientTape() as tape:
            input = tf.cast(item[0], dtype=tf.float32)
            label = tf.cast(item[1], dtype=tf.float32)
            predictions = self.model(input)
            loss = self.lossFun(label, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
    
    @tf.function
    def test_step(self, item):
        input = tf.cast(item[0], dtype=tf.float32)
        label = tf.cast(item[1], dtype=tf.float32)
        predictions = self.model(input)
        t_loss = self.lossFun(label, predictions)
        self.test_loss(t_loss)

    def train(self):
        train_loss_arr = np.array([])
        valid_loss_arr = np.array([])
        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.test_loss.reset_states()

            for item in self.dataset_train:
                self.train_step(item)
                # print(f'Loss: {self.train_loss.result()}')
                # break
            
            for valid_item in self.dataset_valid:
                self.test_step(valid_item)
                break

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '    
                f'Test Loss: {self.test_loss.result()}, '
            )
            train_loss_arr = np.append(train_loss_arr, self.train_loss.result())
            valid_loss_arr = np.append(valid_loss_arr, self.test_loss.result())
        os.makedirs('trained_model', exist_ok=True)
        self.model.save('trained_model/SRCNN.h5')
        np.save('trained_model/train_loss_arr.npy', train_loss_arr)
        np.save('trained_model/valid_loss_arr.npy', valid_loss_arr)

if __name__  == '__main__':
    import argparse
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser(description='SR')
    args = parser.parse_args()
    args.n_colors = 3
    args.epochs = 500
    trianer = Trainer(args=args)
    trianer.train()

