# @Time    : 2020/2/11 21:01
# @Author  : R.Jian
# @NOTE    : 自定义层

import tensorflow as tf


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
        # output = tf.reshape(alpha,[input_shape[0],input_shape[1],-1])
        output = tf.tensordot(output,self.att_keral,[(rank-1),(0)])
        if isfinal:
            return alpha,output
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

        # 【batch，timestep，1】
        output1 = tf.math.divide_no_nan(self.gain,sigma)
        output2 = tf.math.subtract(inputs,mu)
        output3 = tf.math.multiply(output2,output1)
        output = tf.math.add(output3,self.bias)
        # tf.print("output", tf.reduce_sum(output))
        # tf.print()
        return output

class LossLearn_AveragePool(tf.keras.layers.Layer):
    def __init__(self,dense_1_units):
        super(LossLearn_AveragePool,self).__init__()
        self.dense_1 = tf.keras.layers.Dense(dense_1_units,activation="relu")
        "输出的dense加激活函数"
        self.dense_out = tf.keras.layers.Dense(1,activation="relu")

    def build(self, input_shape):
        pass

    def call(self, inputs, mask,**kwargs):
        "[batch,timestep,dim] [batch,timestep]"
        mask_int = tf.cast(mask,tf.float32)
        output = tf.math.multiply(inputs,tf.reshape(mask_int,[tf.shape(mask_int)[0],tf.shape(mask_int)[1],1]))
        sum_output = tf.math.reduce_sum(output,1) #[batch,dim]
        sum_mask = tf.math.reduce_sum(mask_int,-1) #[batch,1]
        output = tf.math.divide(sum_output,tf.reshape(sum_mask,[tf.shape(sum_mask)[0],1])) #[batch,dim]
        output = self.dense_1(output)
        output = self.dense_out(output)
        return output

class LossLearning_Cov(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides,dense_1_units):
        super(LossLearning_Cov,self).__init__()
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self.cov = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding="valid",
            data_format="channels_last",
            activation=tf.keras.activations.relu,
            use_bias=True
        )
        self.dense_1 = tf.keras.layers.Dense(dense_1_units, activation="relu")
        self.dense_out = tf.keras.layers.Dense(1,activation="relu")

    def call(self, inputs,mask, **kwargs):
        '''

        :param inputs:[x,mask] [dim,timestep,dim] float32  [dim,timestep] int32
        :param kwargs:
        :return:
        '''
        input_shape = tf.shape(inputs)
        inputs = tf.transpose(tf.reshape(inputs,[input_shape[0],1,input_shape[1],input_shape[2]]),[0,2,3,1])
        out = self.cov(inputs)
        out = tf.transpose(out,[0,3,1,2])
        out = out[:,:,:self._strides[0]-self._kernel_size[0],:]
        mask = mask[:,self._kernel_size[0]-self._strides[0]:self._strides[0]-self._kernel_size[0]]
        out_shape = tf.shape(out)
        mask_shape = tf.shape(mask)
        mask = tf.reshape(mask,[mask_shape[0],1,mask_shape[1],1])
        mask = tf.tile(mask,[1,out_shape[1],1,out_shape[3]])
        out = tf.math.multiply(out,tf.cast(mask,tf.float32))
        out = tf.reduce_sum(out,[-2])
        out_shape = tf.shape(out)
        out = tf.reshape(out,[out_shape[0],1017])
        out = self.dense_1(out)
        out = self.dense_out(out)
        return out

