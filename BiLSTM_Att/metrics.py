# @Time: 2020/4/22 19:50
# @Author: R.Jian
# @Note: 自定义评价函数

import tensorflow as tf

class F1(tf.keras.metrics.Metric):
    def __init__(self):
        super(F1,self).__init__(name="F1",dtype=tf.float32)
        self.F1 = self.add_weight(
            name="F1",
            shape=(),
            initializer=tf.zeros
        )

    def update_state(self,y_true,y_pred, *args, **kwargs):
        pass

    def result(self):
        pass

    def reset_states(self):
        pass

