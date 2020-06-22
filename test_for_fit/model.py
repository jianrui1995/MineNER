# @Time: 2020/2/27 14:04
# @Author: R.Jian
# @Note: 

import tensorflow as tf
from test_for_fit.dataset import ra
from test_for_fit.Losses import MineLoss
from test_for_fit.Metrics import MineMetric
from test_for_fit.callback import mycallback
from visualdl import LogWriter

class model(tf.keras.Model):
    def __init__(self,):
        super(model,self).__init__()
        self.x1 = tf.keras.layers.Dense(10,activation="relu")
        self.x2 = tf.keras.layers.Dense(10,activation="relu")
        self.pre = tf.keras.layers.Dense(3,activation="softmax")
        self.test = tf.keras.layers.Lambda(self.func)
    @tf.function
    def call(self, inputs, training=None, mask=None):
        out = self.x1(inputs)
        out = self.x2(out)
        out = self.test([out,out])
        out = self.pre(out)
        return [out,out,out]

    def func(self,inputs):
        return inputs[0]

def train():
    re = ra()
    mo = model()
    mo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss=[MineLoss(from_logits=False),MineLoss(from_logits=False),MineLoss(from_logits=False)],
               metrics=[[],[MineMetric()],[MineMetric()]])
    mo.fit(re().batch(4),
           epochs=5,
           validation_data=re().batch(4),
           callbacks=[mycallback()],
           verbose=0)





if __name__ == "__main__":
    train()