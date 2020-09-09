# @Time: 2020/7/6 15:03
# @Author: R.Jian
# @Note: 用于测试模型的正确性，测试结果

from AL_BiLSTM_ATT_LN.model import Model
from preprogram_of_CCKS2019_subtask1.pre_out import OutDataset
import AL_BiLSTM_ATT_LN.HPsetting as setting
import  tensorflow as tf

# x = tf.ones([4,5,300])
# mask = tf.constant([[1,1,1,0,0],[1,1,0,0,0],[1,1,1,1,0],[1,1,1,1,1]])
# mask = tf.cast(mask,tf.bool)
# y = [[[0 for _ in range(13)] for _ in range(5)] for _ in range(4)]
# y = tf.constant(y,tf.int32)
# inout = [x,mask,y]
# mo = Model()
# r = mo(inout,training=False)
# print(r)

mask = tf.constant([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]],tf.int32)
mask = tf.reshape(mask,[2,1,20,1])
mask = tf.tile(mask,[1,3,1,3])
print(mask)