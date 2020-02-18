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
        LSTM_2 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True)
        self.BiLSTM_2 = tf.keras.layers.Bidirectional(LSTM_2, merge_mode="concat")
        LSTM_3 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True)
        self.BiLSTM_2 = tf.keras.layers.Bidirectional(LSTM_3, merge_mode="concat")
        self.Att = Attention(units=p_setting.LABEL_SUM,trainable=True)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        outs_1 = self.BiLSTM_1(inputs,mask=mask)
        outs = self.Att(outs_1,isfinal=True)
        return outs

    def loss(self,outs,y):
        "计算损失函数"
        outs = tf.math.log(outs)
        outs = tf.math.multiply(outs,y)
        loss = tf.math.reduce_sum(outs)
        return -loss

    def test(self,outs):
        "测试需要的操作"
        outs = tf.reshape(outs,[outs.shape[1],outs.shape[2]])
        pre_label = tf.math.argmax(outs,axis=1,output_type=tf.int32)
        return pre_label



def train(num,restore_path=None,epoch=20):
    # num:存储的序号
    # restore_path:为载入模型的路径
    # epoch:为训练的次数
    outdataset = OutDataset(*p_setting.LOAD_PATH)
    bilstm_att = Model()
    op = tf.keras.optimizers.Adam(1e-5)
    ckpt = tf.train.Checkpoint(model=bilstm_att,optimizer=op)#在这个位置更新优化器操作。
    ckptmana = tf.train.CheckpointManager(ckpt,setting.MODEL_PATH,max_to_keep=100,checkpoint_name=setting.MODEL_NAME)
    if restore_path:
        ckpt.restore(restore_path)
    for _ in range(1,epoch+1):
        for data in outdataset().batch(10).take(1):
            with tf.GradientTape() as tape:
                out = bilstm_att(data[0][0],mask=data[0][1])
                loss = bilstm_att.loss(out,data[1][0])
            print(loss)
            grad = tape.gradient(loss,bilstm_att.trainable_variables)
            op.apply_gradients(zip(grad,bilstm_att.trainable_variables))
        print("time ",_," finished")
        if _ % setting.SAVED_EVERY_TIMES == 0:
            print("save")
            num = num + 1
            ckptmana.save(num)

def test(restore_path):
    # restore_path:要载入的文件路径
    outdataset = OutDataset(*p_setting.LOAD_PATH)
    bilstm_att = Model()
    op = tf.keras.optimizers.Adam(1e-5)
    ckpt = tf.train.Checkpoint(model=bilstm_att,optimizer=op)
    ckpt.restore(restore_path)
    for data in outdataset().batch(1).take(1):
        out = bilstm_att(data[0][0], mask=data[0][1])
        result = bilstm_att.test(out)
        print(" ".join([str(i) for i in result.numpy().tolist()]))
if __name__ == "__main__":
    "训练"
    # train(1)
    "测试"
    test(setting.MODEL_PATH+setting.MODEL_NAME+"-2")