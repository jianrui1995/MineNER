# @Time: 2020/4/21 17:45
# @Author: R.Jian
# @Note: model 带有LN机制

import tensorflow as tf
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import AL_BiLSTM_ATT_LN.HPsetting as setting
from AL_BiLSTM_ATT_LN.layers import Attention,LayerNormalization,LossLearn_AveragePool,LossLearning_Cov
from AL_BiLSTM_ATT_LN.losses import *
import preprogram_of_CCKS2019_subtask1.setting as p_setting

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        LSTM_1 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS,return_sequences=True,activation="tanh")
        self.BiLSTM_1 = tf.keras.layers.Bidirectional(LSTM_1,merge_mode="concat")
        self.LN_1 = LayerNormalization()
        self.LL_1 = LossLearning_Cov(setting.FILTERS,setting.KERNEL_SIZE,setting.STRIDES,setting.LL_DENSE_1)
        # self.LL_1 = LossLearn_AveragePool(setting.LL_DENSE_1)

        LSTM_2 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True,activation="tanh")
        self.BiLSTM_2 = tf.keras.layers.Bidirectional(LSTM_2, merge_mode="concat")
        self.LN_2 = LayerNormalization()
        self.LL_2 = LossLearning_Cov(setting.FILTERS, setting.KERNEL_SIZE, setting.STRIDES,setting.LL_DENSE_1)
        # self.LL_2 = LossLearn_AveragePool(setting.LL_DENSE_1)


        LSTM_3 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True,activation="tanh")
        self.BiLSTM_3 = tf.keras.layers.Bidirectional(LSTM_3, merge_mode="concat")
        self.LN_3 = LayerNormalization()
        self.LL_3 = LossLearning_Cov(setting.FILTERS, setting.KERNEL_SIZE, setting.STRIDES,setting.LL_DENSE_1)
        # self.LL_3 = LossLearn_AveragePool(setting.LL_DENSE_1)

        self.Att = Attention(units=p_setting.LABEL_SUM, trainable=True)

        "预测LL的层"
        self.LL_pre_dense = tf.keras.layers.Dense(1,activation=tf.keras.activations.linear)
        # 训练的时候，使用的，用来标记，在训练的时候，是否使用LL
        # 在训练中，要停止更新LL，这设置isUseLL为False
        self.isUseLL = tf.Variable(False,dtype=tf.bool,trainable=False)
        # 在predict阶段，设置ispredict为True
        self.ispredict = tf.Variable(False,dtype=tf.bool,trainable=False) # 当为predict时，变为True

        "LL的边际量"
        "setting中未设置"
        self.theta = tf.constant(setting.THETA,tf.float32)

        self.setdence = setting.LL_DENSE_1*3

    @tf.function()
    def call(self, inputs, training=True, mask=None):
        """

        :param inputs: [x,mask,y]; dtype: x:tf.float32,mask:tf.bool,y:tf.int32
        :param training:
        :param mask:
        :return:
        """
        # 第一层
        output1 = self.BiLSTM_1(inputs[0],mask=inputs[1])
        output1 = self.LN_1(output1)
        output = self.Att(output1, isfinal=False)
        output = tf.concat((output1,output),axis=-1)
        # LL模块
        LL_output_1 = tf.constant(0,tf.float32)
        if training or self.ispredict: #当为validation状态时，training=false
            if self.isUseLL:
                # 新增 tf.stop_gradient()
                LL_output_1 = self.LL_1(tf.stop_gradient(output), inputs[1])
                # LL_output_1 = self.LL_1(output,inputs[1])

        # 第二层
        output1 = self.BiLSTM_2(output,mask=inputs[1])
        output1 = self.LN_2(output1)
        output = self.Att(output1, isfinal=False)
        output = tf.concat((output1,output),axis=-1)
        # LL模块
        LL_output_2 = tf.constant(0,tf.float32)
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                # 新增 tf.stop_gradient()
                LL_output_2 = self.LL_2(tf.stop_gradient(output), inputs[1])
                # LL_output_2 = self.LL_2(output, inputs[1])

        # 第三层
        output1 = self.BiLSTM_3(output,mask=inputs[1])
        output1 = self.LN_3(output1)
        alpha,output = self.Att(output1, isfinal=True)
        output = tf.concat((output1,output),axis=-1)
        # LL模块
        LL_output_3 = tf.constant(0,tf.float32)
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                # 新增 tf.stop_gradient()
                LL_output_3 = self.LL_3(tf.stop_gradient(output), inputs[1])
                # LL_output_3 = self.LL_3(output, inputs[1])


        # LL的预测部分
        LL_output = tf.constant([1],tf.float32)
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                LL_output = tf.concat((LL_output_1,LL_output_2,LL_output_3),axis=-1)
                LL_output = self.LL_pre_dense(tf.reshape(LL_output,[-1,3]))


        output_for_metric = self.prepare_for_metric([alpha,inputs[1]])

        # 计算任务损失
        task_loss = self.en_crass(inputs[2],alpha) #[batch]

        # 计算Loss预测的损失 "
        # 输入是task_loss,LL_output [batch] X2
        LL_loss = tf.zeros(tf.shape(task_loss))
        if training :  # 当为validation状态时，training=false
            if self.isUseLL:
                LL_loss = self.loss_for_LMP(task_loss,LL_output)
        #     else:
        #         LL_loss = tf.zeros(tf.shape(task_loss))
        # else:
        #     LL_loss = tf.zeros(tf.shape(task_loss))

        return [task_loss,LL_loss,LL_output,output_for_metric]


    def prepare_for_metric(self,inputs):
        """
        这个是为方便metric的输入而准备的
        :param inputs:[y_pre,mask] size:[batch,timestep,type_num],[batch,timeste]
        :return: [batch,timestep],dtype:tf.float32
        """
        out = tf.math.argmax(inputs[0],output_type=tf.int32,axis=-1)
        mask = tf.cast(inputs[1],dtype=tf.int32)
        out = tf.math.multiply(out,mask)
        return out

    def en_crass(self, y_true, y_pred):
        """
        使用交叉熵，计算loss
        :param y_true: size:[batch,timestep,type_num] dtype:tf.float32
        :param y_pred: size:[batch,timestep,type_num] dtype:tf.int32
        :return: size:[batch] dtype:tf.float32
        """
        outs = tf.math.log(y_pred,)
        y_true = tf.cast(y_true,tf.float32)
        outs = tf.math.multiply(outs,y_true)
        loss = tf.math.reduce_sum(outs,axis=[-2,-1])
        return -loss

    def loss_for_LMP(self,task_loss,LL_output):
        """
        LMP模块的loss
        :param task_loss: size:[batch]; dtype:tf.float32
        :param LL_output: size:[batch]; dtype:tf.float32
        :return: size: [batch/2]; dtype:tf.float32
        """
        task_loss = tf.reshape(task_loss,[-1,2]) # 位置0 代表i， 位置1 代表j
        LL_output = tf.reshape(LL_output,[-1,2])
        task_max = tf.math.argmax(task_loss,axis=-1,output_type=tf.int32) # [batch/2] 0 代表i大，1代表j大。
        task_max = tf.cast(task_max,tf.bool)
        # 判断li和lj的大小，并返回正负1
        result = tf.where(task_max,x=1.0,y=-1.0)
        # 对i和j预测loss的减法
        LL_subtract = tf.math.subtract(LL_output[:,0:1],LL_output[:,1:]) # [batch,1]
        LL_subtract = tf.reshape(LL_subtract,[-1])
        output =tf.add(tf.multiply(result,LL_subtract),self.theta)
        output = tf.keras.activations.relu(output)
        return output



# if __name__ == "__main__":
#     test_dataset = OutDataset(*setting.LOAD_TEST_PATH)().shuffle(100).batch(setting.BATCH_SIZE,drop_remainder=True).prefetch(3).take(1)
#     train_dataset = OutDataset(*setting.LOAD_NEW_PATH)().batch(setting.BATCH_SIZE,drop_remainder=True).prefetch(3).take(1)
#     model = Model()
#     op = tf.keras.optimizers.Adam(setting.LEARN_RATE)
#     save = Save(save_per_epoch=setting.SAVED_EVERY_TIMES,save_directory=setting.MODEL_PATH_SAVE,save_name=setting.MODEL_NAME_SAVE,restore_path=None)
#     logdir = VisualDL("./result",validation_every_times=setting.SAVED_EVERY_TIMES)
#     model.compile(
#         optimizer=op,
#         loss=[,],
#         metrics=[[],[F1()]]
#     )
#
#     model.fit(
#         x=train_dataset,
#         epochs=setting.EPOCH,
#         verbose=1,
#         initial_epoch=setting.INIT_EPOCH,
#         validation_data=test_dataset,
#         validation_freq=setting.SAVED_EVERY_TIMES
#         # callbacks=[save,logdir]
#     )






