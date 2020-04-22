# @Time: 2020/4/21 17:45
# @Author: R.Jian
# @Note: model 带有LN机制

import tensorflow as tf
import BiLSTM_Att.setting as setting
from BiLSTM_Att.myself_layer import *
from preprogram_of_CCKS2019_subtask1.pre_out import OutDataset

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        LSTM_1 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS,return_sequences=True)
        self.BiLSTM_1 = tf.keras.layers.Bidirectional(LSTM_1,merge_mode="concat")
        self.LN_1 = LayerNormalization()

        LSTM_2 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True)
        self.BiLSTM_2 = tf.keras.layers.Bidirectional(LSTM_2, merge_mode="concat")
        self.LN_2 = LayerNormalization()

        LSTM_3 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True)
        self.BiLSTM_3 = tf.keras.layers.Bidirectional(LSTM_3, merge_mode="concat")
        self.LN_3 = LayerNormalization()

        self.Att = Attention(units=p_setting.LABEL_SUM, trainable=True)

    @tf.function()
    def call(self, inputs, training=None, mask=None):
        # 确定dataset的输入格式符合fit的要求。
        output = self.BiLSTM_1(inputs[0],mask=inputs[1])
        output = self.LN_1(output)
        output = self.Att(output, isfinal=False)

        output = self.BiLSTM_1(output,mask=inputs[1])
        output = self.LN_1(output)
        output = self.Att(output, isfinal=False)

        output = self.BiLSTM_1(output,mask=inputs[1])
        output = self.LN_1(output)
        output = self.Att(output, isfinal=True)

        return output

if __name__ == "__main__":
    out = OutDataset(*setting.LOAD_NEW_PATH)
    model = Model()