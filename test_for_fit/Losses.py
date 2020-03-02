# @Time: 2020/2/28 15:09
# @Author: R.Jian
# @Note: 自定义损失函数

import tensorflow as tf

class MineLoss(tf.keras.losses.Loss):
    def __init__(self,from_logits):
        super(MineLoss,self).__init__()
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=self.from_logits)
        return loss

