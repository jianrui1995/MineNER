# @Time: 2020/4/22 19:41
# @Author: R.Jian
# @Note: 自定义的损失函数。

import tensorflow as tf

class En_Cross(tf.keras.losses.Loss):
    def __init__(self):
        super(En_Cross,self).__init__()

    def call(self, y_true, y_pred):
        outs = tf.math.log(y_pred)
        outs = tf.math.multiply(outs,y_true)
        loss = tf.math.reduce_sum(outs)
        return -loss


