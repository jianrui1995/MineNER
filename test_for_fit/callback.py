# @Time: 2020/6/18 14:29
# @Author: R.Jian
# @Note: 

import tensorflow as tf
from visualdl import LogWriter

class mycallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(mycallback,self).__init__()


    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        tf.print("Apoch",epoch)

    def on_train_end(self, logs=None):
        pass

