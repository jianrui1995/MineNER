# @Time: 2020/7/13 15:39
# @Author: R.Jian
# @Note: 子层所用到的层。

import tensorflow as tf
import numpy as np
import BERT_LAN.HPsetting as HP

class Create_Heads(tf.keras.layers.Layer):
    def __init__(self,heads_num):
        super(Create_Heads,self).__init__()
        self.heads_num = heads_num

    def build(self, input_shape):
        # input size:[dim,timestep,dim]
        self.query = [self.add_weight(
            shape=[input_shape[-1],input_shape[-1]//self.heads_num],
            dtype=tf.float32,
            trainable=True,
            name="query_{}".format(i)
        ) for i in range(self.heads_num)]
        self.key = [self.add_weight(
            shape=[input_shape[-1],input_shape[-1]//self.heads_num],
            dtype=tf.float32,
            trainable=True,
            name="key_{}".format(i)
        ) for i in range(self.heads_num)]
        self.value = [self.add_weight(
            shape=[input_shape[-1],input_shape[-1]//self.heads_num],
            dtype=tf.float32,
            trainable=True,
            name="value_{}".format(i)
        ) for i in range(self.heads_num)]

    def call(self, inputs, **kwargs):
        '''

        :param inputs: size:[batch,timestep,dim] dtype:tf.float32
        :param kwargs:
        :return:
        '''
        heads_query = [tf.matmul(inputs, query) for query in self.query]
        heads_key = [tf.matmul(inputs, key) for key in self.key]
        heads_value = [tf.matmul(inputs, value) for value in self.value]
        return heads_query, heads_key, heads_value

class Self_Attention(tf.keras.layers.Layer):
    def __init__(self,mask_num):
        super(Self_Attention,self).__init__()
        self._mask_num = mask_num

    def build(self, input_shape):
        pass

    def call(self, inputs,mask=None, **kwargs):
        '''

        :param inputs:[q,k,v] 每个[batch,timestep,dim] dtype:tf.float32
        :param mask:[batch,tiemstep] tf.int32
        :param kwargs:
        :return:
        '''
        queries,keies,values = inputs
        keies_shape = tf.shape(keies)
        keies = tf.transpose(keies,[0,2,1])
        outputs = tf.matmul(queries,keies)
        outputs = tf.math.divide(outputs,tf.math.sqrt(tf.cast(keies_shape[2],dtype=tf.float32))) # [batch,timestep,timestep]
        # 获得和outputs同大小的ones矩阵
        _mask_for_outputs = tf.linalg.band_part(tf.ones(tf.shape(outputs),dtype=tf.int32),self._mask_num,self._mask_num) # [batch,timestep,timestep]
        # mask = tf.cast(mask,dtype=tf.float32) # [batch,timestep] float32

        # 对output和_mask_for_outputs进行有效值的修剪。
        mask_shape = tf.shape(mask)
        _mask = tf.reshape(mask,[mask_shape[0],1,mask_shape[1]]) # [batch,1,timestep]
        _mask_for_outputs_1 = tf.math.subtract(tf.ones(tf.shape(outputs),dtype=tf.int32),tf.transpose(_mask,[0,2,1]))
        _mask_for_outputs_1 = tf.math.multiply(_mask_for_outputs_1,_mask)
        # outputs = tf.math.multiply(outputs,_mask) # [batch,timestep,timestep]  横坐标上超出mask的部分变为了0

        _mask_for_outputs = tf.math.multiply(_mask_for_outputs,_mask) # [batch,timestep,timestep]  横坐标上超出mask的部分变为了0
        _mask_for_outputs = tf.math.add(_mask_for_outputs,_mask_for_outputs_1)
        # 将_mask_for_outputs装换为bool类型，用于tf.where
        _mask_for_outputs = tf.cast(_mask_for_outputs,dtype=tf.bool)
        # mask对应的值，其输出结果是0
        outputs = tf.where(_mask_for_outputs,outputs,-np.inf)

        outputs = tf.keras.activations.softmax(outputs,-1)
        # 用查询结果和values相乘求和。
        outputs = tf.matmul(outputs,values)
        return outputs

class Merge_Heads(tf.keras.layers.Layer):
    def __init__(self):
        super(Merge_Heads,self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[input_shape[-1],input_shape[-1]],
            dtype=tf.float32,
            trainable=True,
            name="merge_weigth"
        )

    def call(self, inputs, **kwargs):
        '''

        :param inputs:
        :param kwargs:
        :return:
        '''
        outputs = tf.matmul(inputs,self.w)
        return outputs

class Full_Connercted_Netword(tf.keras.layers.Layer):
    def __init__(self,hidden_dim = HP.FULL_CONNETECTED_HIDDEN_DIM,word_embedding_dim = HP.WORD_EMBEDDING_DIM):
        super(Full_Connercted_Netword,self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(hidden_dim,activation=tf.keras.activations.relu)
        self.output_layer = tf.keras.layers.Dense(word_embedding_dim,activation=None)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        '''

        :param inputs:[batch,timestep,dim],dtype=tf.float32
        :param kwargs:
        :return:
        '''
        outputs = self.hidden_layer(inputs)
        outputs = self.output_layer(outputs)
        return outputs

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 activity_regularizer=None,
                 activation = tf.keras.activations.tanh,
                 **kwargs
                 ):
        super(LayerNormalization,self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            **kwargs
        )

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=[input_shape[-1]],
            dtype=tf.float32,
            trainable=True
        )
        self.gain = self.add_weight(
            name="gain",
            shape=[input_shape[-1]],
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs, **kwargs):
        '''

        :param inputs: [batch,timestep,dim] dtype:tf.float32
        :param kwargs:
        :return:
        '''
        # 求平均
        mu = tf.math.reduce_mean(inputs,-1)
        mu_shape = tf.shape(mu)
        mu = tf.reshape(mu,[mu_shape[0],mu_shape[1],1])
        # tf.print("mu",tf.reduce_sum(mu))
        # 算方差
        sigma = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.subtract(inputs,mu)),-1))
        sigma_shape = tf.shape(sigma)
        sigma = tf.reshape(sigma,[sigma_shape[0],sigma_shape[1],1])
        # tf.print("sigma", tf.reduce_sum(sigma))

        # [batch，timestep，1]
        output1 = tf.math.divide_no_nan(self.gain,sigma)
        output2 = tf.math.subtract(inputs,mu)
        output3 = tf.math.multiply(output2,output1)
        output = tf.math.add(output3,self.bias)
        # tf.print("output", tf.reduce_sum(output))
        # tf.print()
        return output