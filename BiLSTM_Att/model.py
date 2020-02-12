# @Time    : 2020/2/8 16:59
# @Author  : R.Jian
# @NOTE    : 

import tensorflow as tf

import BiLSTM_Att.setting as setting
from preprogram_of_CCKS2019_subtask1.pre_out import OutDataset
import preprogram_of_CCKS2019_subtask1.setting as p_setting
from BiLSTM_Att.myself_layer import Attention

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        LSTM_1 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS,return_sequences=True)
        self.BiLSTM_1 = tf.keras.layers.Bidirectional(LSTM_1,merge_mode="concat")
        self.Att = Attention(units=p_setting.LABEL_SUM,trainable=True)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outs_1 = self.BiLSTM_1(inputs,mask=mask)
        outs = self.Att(outs_1,isfinal=True)
        return outs

    def loss(self,outs,y):
        outs = tf.math.log(outs)
        outs = tf.math.multiply(outs,y)
        print(outs)
        return outs


def train(num,restore_path=None,epoch=6):
    outdataset = OutDataset(*p_setting.LOAD_PATH)
    bilstm_att = Model()
    ckpt = tf.train.Checkpoint(model=bilstm_att)#在这个位置更新优化器操作。
    ckptmana = tf.train.CheckpointManager(ckpt,"../model/BiLSTM_Att/",max_to_keep=100,checkpoint_name="bilstm_att")
    if restore_path:
        ckpt.restore(restore_path)
    for data in outdataset().batch(5).take(epoch):
        out = bilstm_att(data[0][0],mask=data[0][1])
        out = bilstm_att.loss(out,data[1][0])
        # print(out.shape)
    ckptmana.save(num)


if __name__ == "__main__":
    train(1)