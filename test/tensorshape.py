# @Time    : 2020/2/11 12:39
# @Author  : R.Jian
# @NOTE    : 测试tensorshape


import numpy as np
import time
import tensorflow as tf

a = tf.ones([2,2,2])
b = tf.constant([1.0,2.0])
print(tf.math.multiply(a,b)+b)