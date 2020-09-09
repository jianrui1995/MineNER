# @Time: 2020/7/24 10:52
# @Author: R.Jian
# @Note: 

import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(2.0)
z = tf.Variable(3.0)
w = tf.Variable(3.0)

with tf.GradientTape() as t:
  out = x**2
  out1 = y**2 + out
  # out = z**2 + tf.stop_gradient(out1)
  out = z**2 + out1
  out2 = out1 + w**2

grad = t.gradient([out,out2], {'x': x, 'y': y,'z':z,'w':w})

print('dx:', grad['x'])  # 2*x => 4
print('dy:', grad['y'])
print('dz:', grad['z'])
print('dw:', grad['w'])