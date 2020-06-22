# @Time: 2020/6/18 14:29
# @Author: R.Jian
# @Note: 

import tensorflow as tf
from visualdl import LogWriter

class mycallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(mycallback,self).__init__()
        self.total_loss = 0.0

    def on_batch_end(self, batch, logs=None):
        print("batch")
        print(logs)

    def on_epoch_end(self, epoch, logs=None):
        print("epoch")
        print(logs)

    def on_train_end(self, logs=None):
        pass

