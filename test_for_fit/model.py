# @Time: 2020/2/27 14:04
# @Author: R.Jian
# @Note: 
import threading
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import tensorflow as tf
from test_for_fit.dataset import ra
from test_for_fit.Losses import MineLoss
from test_for_fit.Metrics import MineMetric,OMineMetric
from test_for_fit.callback import mycallback
from test_for_fit.optimizers import mine_op
class model(tf.keras.Model):
    def __init__(self,):
        super(model,self).__init__()
        self.x1 = tf.keras.layers.Dense(10,activation="relu",name="a")
        self.x2 = tf.keras.layers.Dense(10,activation="relu")
        self.pre = tf.keras.layers.Dense(3,activation="softmax",name="b")
        self.test = tf.keras.layers.Lambda(self.func)
        self.isuse = tf.Variable(True,dtype=tf.bool,trainable=False)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        out = self.x1(inputs)
        out = self.pre(out)
        return [out,out,out]

    def func(self,inputs):
        return inputs[0]



def train():
    re = ra()
    mo = model()

    mo.compile(optimizer=mine_op(0.001),
               loss=[None,MineLoss(from_logits=False,name="B_loss"),MineLoss(from_logits=False,name="C_loss")],
               metrics=[[MineMetric()],[MineMetric(),OMineMetric()],[MineMetric()]])

    mo.fit(re().batch(4).take(1),
           epochs=3,
           # validation_data=re().batch(4),
           callbacks=[mycallback()],
           verbose=1,
           initial_epoch=0
           )

def predict():
    re = ra()
    mo = model()
    a =  mo.predict(
        x=re().batch(4).take(3),
        callbacks=[mycallback()],
    )

    # 针对某一层的 变量进行改变。
    # mo.x1.trainable = False
    # print(mo.x1.non_trainable_variables)
    # mo.x1.trainable = True
    # print(mo.x1.non_trainable_variables)
    # print(mo.trainable_variables)



if __name__ == "__main__":
    train()
    # predict()