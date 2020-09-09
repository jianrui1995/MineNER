# @Time: 2020/6/30 15:09
# @Author: R.Jian
# @Note: 一些函数的功能性测试
#
import  tensorflow as tf


class LossLearning_Cov(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides):
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
        self.dense_out = tf.keras.layers.Dense(1,activation="relu")
    @tf.function
    def call(self, inputs, **kwargs):
        '''

        :param inputs:[x,mask] [dim,timestep,dim] float32  [dim,timestep] int32
        :param kwargs:
        :return:
        '''
        out = self.cov(inputs[0])
        out = tf.transpose(out,[0,3,1,2])
        out = out[:,:,:self._strides[0]-self._kernel_size[0],:]
        mask = inputs[1][:,self._kernel_size[0]-self._strides[0]:self._strides[0]-self._kernel_size[0]]
        out_shape = tf.shape(out)
        mask_shape = tf.shape(mask)
        mask = tf.reshape(mask,[mask_shape[0],1,mask_shape[1],1])
        mask = tf.tile(mask,[1,out_shape[1],1,out_shape[3]])
        out = tf.math.multiply(out,tf.cast(mask,tf.float32))
        out = tf.reduce_sum(out,[-2])
        out = tf.reshape(out,[out_shape[0],-1])
        out = self.dense_out(out)
        return out


if __name__ == "__main__":
    _filters = 3
    _kernel_size = [5, 8]
    _strides = [1, 3]
    input = tf.ones([2,20,512,1],dtype=tf.float32)
    mask = tf.constant([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]],tf.int32)
    lay = LossLearning_Cov(_filters,_kernel_size,_strides)
    print(lay([input,mask]))
