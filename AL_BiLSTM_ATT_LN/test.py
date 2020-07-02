# @Time: 2020/6/30 15:09
# @Author: R.Jian
# @Note: 一些函数的功能性测试

import  tensorflow as tf

a = tf.ones([3,3,3],dtype=tf.int32)
b = tf.constant([[1,1,0],[1,0,0],[1,1,1]])


"测试"
print(b[:,2:])

"测试"
# c = tf.multiply(a,b)
# d = tf.math.reduce_sum(c,1)
# print(d)
# e = tf.math.reduce_sum(b,-1)
# print(e)
# f = tf.math.divide(d,tf.reshape(e,[tf.shape(e)[0],1]))
# print(f)

