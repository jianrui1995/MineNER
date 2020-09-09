# @Time: 2020/6/18 14:29
# @Author: R.Jian
# @Note: 

import tensorflow as tf
from visualdl import LogWriter

class mycallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(mycallback,self).__init__()

    # def on_epoch_end(self, epoch, logs=None):
    #     pass
    # def set_model(self, model):
    #     self.model = model

    def on_predict_begin(self, logs=None):
        print("--on_predict_begin--")
        print(logs)
        print()

    def on_predict_end(self, logs=None):
        print("--on_predict_end--")
        print(logs)
        print()

    def on_predict_batch_begin(self, batch, logs=None):
        print("--on_predict_batch_begin--")
        print(logs)
        print()

    def on_predict_batch_end(self, batch, logs=None):
        print("--on_predict_batch_end--")
        print(logs)
        print()