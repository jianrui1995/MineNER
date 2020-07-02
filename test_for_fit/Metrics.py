# @Time: 2020/2/28 14:58
# @Author: R.Jian
# @Note: 自定义指标

import tensorflow as tf

class MineMetric(tf.keras.metrics.Metric):
    def __init__(self,name="minemetirc"):
        super(MineMetric,self).__init__(name=name)
        self.mine = self.add_weight(shape=(),initializer=tf.zeros,name="AAA")
        print(self.mine)
        self.one = self.add_weight(shape=(),initializer=tf.ones,name="BBB")

    def update_state(self, y_true,y_pre,*args, **kwargs):
        self.mine.assign_add(self.one)

    def result(self):
        return self.mine.assign_add(self.one)

    def reset_states(self):
        self.mine.assign(0.0)

class OMineMetric(tf.keras.metrics.Metric):
    def __init__(self,name="ominemetirc"):
        super(OMineMetric,self).__init__(name=name)
        self.mine = self.add_weight(shape=(),initializer=tf.zeros,name="AAA")
        print(self.mine)
        self.one = self.add_weight(shape=(),initializer=tf.ones,name="BBB")

    def update_state(self, y_true,y_pre,*args, **kwargs):
        self.mine.assign_add(self.one)

    def result(self):
        return self.mine.assign_add(self.one)

    def reset_states(self):
        self.mine.assign(0.0)