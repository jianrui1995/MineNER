# @Time: 2020/4/21 17:45
# @Author: R.Jian
# @Note: model 带有LN机制

import tensorflow as tf
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import AL_BiLSTM_ATT_LN.setting_model_LN as setting
from AL_BiLSTM_ATT_LN.layers import Attention,LayerNormalization,LossLearn_AveragePool
from preprogram_of_CCKS2019_subtask1.pre_out import OutDataset
from AL_BiLSTM_ATT_LN.losses import Loss_Return_0
import preprogram_of_CCKS2019_subtask1.setting as p_setting
from AL_BiLSTM_ATT_LN.metrics import F1
from AL_BiLSTM_ATT_LN.callbacks import Save,VisualDL

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        LSTM_1 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS,return_sequences=True,activation="tanh")
        self.BiLSTM_1 = tf.keras.layers.Bidirectional(LSTM_1,merge_mode="concat")
        self.LN_1 = LayerNormalization()
        self.LL_1 = LossLearn_AveragePool(setting.LL_DENSE_1)

        LSTM_2 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True,activation="tanh")
        self.BiLSTM_2 = tf.keras.layers.Bidirectional(LSTM_2, merge_mode="concat")
        self.LN_2 = LayerNormalization()
        self.LL_2 = LossLearn_AveragePool(setting.LL_DENSE_1)


        LSTM_3 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True,activation="tanh")
        self.BiLSTM_3 = tf.keras.layers.Bidirectional(LSTM_3, merge_mode="concat")
        self.LN_3 = LayerNormalization()
        self.LL_3 = LossLearn_AveragePool(setting.LL_DENSE_1)

        self.Att = Attention(units=p_setting.LABEL_SUM, trainable=True)
        self.Prepare_For_Metric = tf.keras.layers.Lambda(self.prepare_for_metric)

        "预测LL的层"
        self.LL_pre_dense = tf.keras.layers.Dense(1)
        # 训练的时候，使用的，用来标记，在训练的时候，是否使用LL
        # 在训练中，要停止更新LL，这设置isUseLL为False
        self.isUseLL = True
        # 在predict阶段，设置ispredict为True
        self.ispredict = False # 当为predict时，变为True

        "LL的边际量"
        "setting中未设置"
        self.theta = tf.constant(setting.THETA,tf.float32)

    @tf.function()
    def call(self, inputs, training=True, mask=None):
        """
        inputs的格式[x,mask,y]
        """

        "第一层"
        output1 = self.BiLSTM_1(inputs[0],mask=inputs[1])
        output1 = self.LN_1(output1)
        output = self.Att(output1, isfinal=False)
        output = tf.concat((output1,output),axis=-1)
        "LL模块"
        if training or self.ispredict: #当为validation状态时，training=false
            if self.isUseLL:
                LL_output_1 = self.LL_1(output,inputs[1])

        "第二层"
        output1 = self.BiLSTM_2(output,mask=inputs[1])
        output1 = self.LN_2(output1)
        output = self.Att(output1, isfinal=False)
        output = tf.concat((output1,output),axis=-1)
        "LL模块"
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                LL_output_2 = self.LL_2(output, inputs[1])

        "第三层"
        output1 = self.BiLSTM_3(output,mask=inputs[1])
        output1 = self.LN_3(output1)
        alpha,output = self.Att(output1, isfinal=True)
        output = tf.concat((output1,output))
        "LL模块"
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                LL_output_3 = self.LL_3(output, inputs[1])


        "LL的预测部分"
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                LL_output = tf.concat((LL_output_1,LL_output_2,LL_output_3))
                LL_output = self.LL_pre_dense(LL_output)


        "计算任务损失"
        task_loss = self.en_crass(inputs[2],alpha) #[batch]

        "计算Loss预测的损失" "输入是task_loss,LL_output" #[batch] X2
        if training or self.ispredict:  # 当为validation状态时，training=false
            if self.isUseLL:
                LL_loss = self.loss_for_LMP(task_loss,LL_output)
            else:
                LL_loss = 0
        else:
            LL_loss = 0


        output_for_metric = self.Prepare_For_Metric([alpha,inputs[1]])

        return [task_loss,self.LL_loss,output_for_metric]
    # 两个输出 分别是loss

    def prepare_for_metric(self,inputs):
        "这个是为方便metric的输入而准备的"
        out = tf.math.argmax(inputs[0],output_type=tf.int32,axis=-1)
        mask = tf.cast(inputs[1],dtype=tf.int32)
        return tf.math.multiply(out,mask)

    def en_crass(self, y_true, y_pred):
        outs = tf.math.log(y_pred)
        outs = tf.math.multiply(outs,y_true)
        loss = tf.math.reduce_sum(outs,axis=-1)
        return -loss

    def loss_for_LMP(self,task_loss,LL_output):
        task_loss = tf.reshape(task_loss,[-1,2]) # 位置0 代表i， 位置1 代表j
        LL_output = tf.reshape(LL_output,[-1,2])
        task_max = tf.math.argmax(task_loss,axis=-1) # [batch/2] 0 代表i大，1代表j大。
        task_max = tf.cast(task_max,tf.bool)
        "判断li和lj的大小，并返回正负1"
        result = tf.where(task_max,x=-1,y=1)
        "对i和j预测loss的减法"
        LL_subtract = tf.math.subtract(LL_output[:,0:1],LL_output[:,1:])
        output = tf.keras.activations.relu(tf.add(tf.multiply(result,LL_subtract),self.theta))



if __name__ == "__main__":
    test_dataset = OutDataset(*setting.LOAD_TEST_PATH)().shuffle(100).batch(setting.BATCH_SIZE,drop_remainder=True).prefetch(3).take(1)
    train_dataset = OutDataset(*setting.LOAD_NEW_PATH)().batch(setting.BATCH_SIZE,drop_remainder=True).prefetch(3).take(1)
    model = Model()
    op = tf.keras.optimizers.Adam(setting.LEARN_RATE)
    save = Save(save_per_epoch=setting.SAVED_EVERY_TIMES,save_directory=setting.MODEL_PATH_SAVE,save_name=setting.MODEL_NAME_SAVE,restore_path=None)
    logdir = VisualDL("./result",validation_every_times=setting.SAVED_EVERY_TIMES)
    model.compile(
        optimizer=op,
        loss=[,],
        metrics=[[],[F1()]]
    )

    model.fit(
        x=train_dataset,
        epochs=setting.EPOCH,
        verbose=1,
        initial_epoch=setting.INIT_EPOCH,
        validation_data=test_dataset,
        validation_freq=setting.SAVED_EVERY_TIMES
        # callbacks=[save,logdir]
    )






