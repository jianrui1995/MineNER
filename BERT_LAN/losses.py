# @Time: 2020/7/22 20:41
# @Author: R.Jian
# @Note: 

import tensorflow as tf

class En_Cross(tf.keras.losses.Loss):
    def __init__(self):
        super(En_Cross,self).__init__(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def call(self, y_true, y_pred):
        '''

        :param y_true: [batch,timetep,labelnum]
        :param y_pred: [batch,timetep,labelnum]
        :return:
        '''
        y_pred = tf.math.log(y_pred)
        y_true = tf.cast(y_true,tf.float32)
        loss = tf.math.multiply(y_true,y_pred)
        loss = tf.reduce_sum(loss,[-2,-1])
        return -loss