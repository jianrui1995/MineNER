# @Time: 2020/7/13 16:18
# @Author: R.Jian
# @Note:

import tensorflow as tf
from BERT_LAN.sublayers import *
import BERT_LAN.HPsetting as HP

class Encoder(tf.keras.layers.Layer):
    def __init__(self,multi_heads_num=HP.HEADS_NUM,mask_num=HP.MASK_NUM):
        super(Encoder,self).__init__()
        self.create_heads = Create_Heads(multi_heads_num)
        self.self_attention = Self_Attention(mask_num)
        self.merge_heads = Merge_Heads()
        # self.full_connected_network = Full_Connercted_Netword(hidden_dim=HP.FULL_CONNETECTED_HIDDEN_DIM,word_embedding_dim=HP.WORD_EMBEDDING_DIM)
        # self.ln_1 = LayerNormalization()
        # self.ln_2 = LayerNormalization()
        # self.dropout = tf.keras.layers.Dropout(HP.DROP_OUT_RATE)


    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        '''

        :param inputs:[x,mask] x:size:[batch,timestep,dim],dtype:tf.float32  mask:size[batch,timestep],dtype:tf.int32
        :param kwargs:
        :return:
        '''
        heads_query,heads_key,heads_value = self.create_heads(inputs[0])
        query_key_value = list(zip(heads_query,heads_key,heads_value))
        outputs = [self.self_attention(q_k_v,inputs[1]) for q_k_v in query_key_value]
        outputs = tf.concat(outputs,-1)
        self_attention_outputs = self.merge_heads(outputs)
        outputs = self_attention_outputs
        # # add&normal
        # outputs = self.dropout(self_attention_outputs,kwargs["training"])
        # outputs = tf.add(inputs[0],outputs)
        # outputs_1 = self.ln_1(outputs)
        #
        # outputs = self.full_connected_network(outputs_1)
        #
        # # add&normal
        # outputs = self.dropout(outputs,kwargs["training"])
        # outputs = tf.add(outputs_1,outputs)
        # outputs = self.ln_2(outputs)

        return outputs


class LabelAttention(tf.keras.layers.Layer):
    def __init__(self,
                 units=None,# label类型的数量
                 **kwargs
                 ):
        super(LabelAttention,self).__init__()
        self.units = units

    def build(self, input_shape):
        self.att_keral=self.add_weight(
            name="att_keral",
            shape=[self.units,input_shape[-1]],
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs,isfinal=False, **kwargs):
        input_shape = inputs.shape
        # rank = len(input_shape)
        output = tf.tensordot(inputs,self.att_keral,axes=[(-1),(1)]) # [batch,timestep,dim],[labelnum,dim] = [batch,timestep,labelnum]
        # output = tf.reshape(output,[-1,output.shape[-1]])
        sqe = tf.sqrt(tf.cast(input_shape[-1],tf.float32))
        output = tf.math.multiply(output,tf.math.reciprocal(sqe))
        alpha = tf.keras.activations.softmax(output,axis=-1)
        # output = tf.reshape(alpha,[input_shape[0],input_shape[1],-1])
        output = tf.tensordot(output,self.att_keral,[(-1),(0)])
        if isfinal:
            return alpha
        return output