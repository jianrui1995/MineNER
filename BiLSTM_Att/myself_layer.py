# @Time    : 2020/2/11 21:01
# @Author  : R.Jian
# @NOTE    : 自定义层

import tensorflow as tf

import BiLSTM_Att.setting as setting
import preprogram_of_CCKS2019_subtask1.setting as p_setting

class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 activity_regularizer=None,
                 units=None,# label类型的数量
                 **kwargs
                 ):
        super(Attention,self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),**kwargs
        )
        self.units = units

    def build(self, input_shape):
        tensor_shape = tf.TensorShape(input_shape)
        self.att_keral=self.add_weight(
            "att_keral",
            shape=[self.units,tensor_shape[-1]],
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs,isfinal=False, **kwargs):
        input_shape = inputs.shape
        rank = len(input_shape)
        output = tf.tensordot(inputs,self.att_keral,axes=[(rank-1),(1)])
        # output = tf.reshape(output,[-1,output.shape[-1]])
        sqe = tf.sqrt(tf.cast(2*input_shape[-1],tf.float32))
        output = tf.math.multiply(output,tf.math.reciprocal(sqe))
        alpha = tf.keras.activations.softmax(output,axis=-1)
        if isfinal:
            return alpha
        # output = tf.reshape(alpha,[input_shape[0],input_shape[1],-1])
        output = tf.tensordot(output,self.att_keral,[(rank-1),(0)])
        return output

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
        self.activation = activation

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=(),
            dtype=tf.float32,
            trainable=True
        )
        self.gain = self.add_weight(
            name="gain",
            shape=(),
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs, **kwargs):
        mu = tf.math.reduce_mean(inputs,-1)
        mu_shape = tf.shape(mu)
        mu = tf.reshape(mu,[mu_shape[0],mu_shape[1],1])

        sigma = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.math.subtract(inputs,mu)),-1))
        sigma_shape = tf.shape(mu)
        sigma = tf.reshape(sigma,[sigma_shape[0],sigma_shape[1],1])

        output1 = tf.math.divide_no_nan(self.gain,sigma)
        output2 = tf.math.subtract(inputs,mu)
        output3 = tf.math.multiply(output2,output1)
        output = tf.math.add(output3,self.bias)
        output = self.activation(output)
        return output