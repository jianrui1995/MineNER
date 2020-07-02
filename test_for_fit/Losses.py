# @Time: 2020/2/28 15:09
# @Author: R.Jian
# @Note: 自定义损失函数

import tensorflow as tf

class MineLoss(tf.keras.losses.Loss):
    def __init__(self,from_logits,name):
        super(MineLoss,self).__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true,y_pred,from_logits=self.from_logits)
        return loss

if __name__ == "__main__":
    y_pre = [[0.1,0.2,0.3],[0.1,0.2,0.3]]
    y_ture = [[0,1,0],[0,1,0]]
    lo = MineLoss(False)
    print(lo(y_ture,y_pre))
