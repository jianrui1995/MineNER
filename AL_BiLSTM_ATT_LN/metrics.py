# @Time: 2020/4/22 19:50
# @Author: R.Jian
# @Note: 自定义评价函数

import tensorflow as tf

class F1(tf.keras.metrics.Metric):
    def __init__(self,name="F1",dtype=tf.float32):
        super(F1,self).__init__(name=name,dtype=dtype)
        self.corrcet = self.add_weight(
            name="correct",
            shape=(),
            initializer=tf.zeros,
            dtype=tf.float32
        )
        self.pre = self.add_weight(
            name="pre_correct",
            shape=(),
            initializer=tf.ones,
            dtype=tf.float32
        )
        self.rel = self.add_weight(
            name="rel_correct",
            shape=(),
            initializer=tf.zeros,
            dtype=tf.float32
        )

    def update_state(self,y_true,y_pred, *args, **kwargs):
        y_true_0 = tf.math.argmax(y_true,axis=-1,output_type=tf.int32)
        self.rel.assign_add(tf.math.count_nonzero(y_true_0,dtype=tf.float32))

        y_pred_0 = y_pred
        self.pre.assign_add(tf.math.count_nonzero(y_pred_0,dtype=tf.float32))

        y_subtract = tf.math.subtract(y_true_0,y_pred_0)
        y_add = tf.math.add(y_pred_0,y_true_0)

        y_subtract_nozeors = tf.math.count_nonzero(y_subtract,dtype=tf.float32)
        y_add_nonzeros = tf.math.count_nonzero(y_add,dtype=tf.float32)

        self.corrcet.assign_add(tf.math.subtract(y_add_nonzeros,y_subtract_nozeors))


    def result(self):
        precision = tf.math.divide(tf.cast(self.corrcet,dtype=tf.float32),tf.cast(self.pre,dtype=tf.float32))
        recall = tf.math.divide(tf.cast(self.corrcet,dtype=tf.float32),tf.cast(self.rel,dtype=tf.float32))
        f1 = (2*precision*recall)/(precision+recall+0.001)
        return f1

    def reset_states(self):
        self.rel.assign(0.0)
        self.corrcet.assign(0.0)
        self.pre.assign(1.0)

