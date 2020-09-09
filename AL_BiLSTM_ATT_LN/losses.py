# @Time: 2020/4/22 19:41
# @Author: R.Jian
# @Note: 自定义的损失函数。

import tensorflow as tf

class LossForTask(tf.keras.losses.Loss):
    def __init__(self):
        super(LossForTask,self).__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):
        return y_pred

class LossForLMP(tf.keras.losses.Loss):
    def __init__(self):
        super(LossForLMP,self).__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self,y_true,y_pred):
        return y_pred

class Loss_Return_0(tf.keras.losses.Loss):
    def __init__(self):
        super(Loss_Return_0,self).__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self,y_true,y_pred):
        return 0.0
