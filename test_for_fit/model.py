# @Time: 2020/2/27 14:04
# @Author: R.Jian
# @Note: 

import tensorflow as tf
from test_for_fit.dataset import ra
from test_for_fit.Losses import MineLoss
from test_for_fit.Metrics import MineMetric

class model(tf.keras.Model):
    def __init__(self):
        super(model,self).__init__()
        self.x1 = tf.keras.layers.Dense(10,activation="relu")
        self.x2 = tf.keras.layers.Dense(10,activation="relu")
        self.pre = tf.keras.layers.Dense(3,activation="softmax")
    @tf.function
    def call(self, inputs, training=None, mask=None):
        out = self.x1(inputs)
        out = self.x2(out)
        out = self.pre(out)
        return out

def train():
    re = ra()
    mo = model()
    mo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss=MineLoss(from_logits=False),
               metrics=[MineMetric()])
    mo.fit(re().batch(4),epochs=5)





if __name__ == "__main__":
    train()