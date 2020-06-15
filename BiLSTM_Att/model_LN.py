# @Time: 2020/4/21 17:45
# @Author: R.Jian
# @Note: model 带有LN机制

import tensorflow as tf
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import BiLSTM_Att.setting_model_LN as setting
from BiLSTM_Att.layers import Attention,LayerNormalization
from preprogram_of_CCKS2019_subtask1.pre_out import OutDataset
from BiLSTM_Att.losses import En_Cross,Loss_Return_0
import preprogram_of_CCKS2019_subtask1.setting as p_setting
from BiLSTM_Att.metrics import F1
from BiLSTM_Att.callbacks import Save

class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        LSTM_1 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS,return_sequences=True,activation=tf.keras.activations.linear)
        self.BiLSTM_1 = tf.keras.layers.Bidirectional(LSTM_1,merge_mode="concat")
        self.LN_1 = LayerNormalization()

        LSTM_2 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True,activation=tf.keras.activations.linear)
        self.BiLSTM_2 = tf.keras.layers.Bidirectional(LSTM_2, merge_mode="concat")
        self.LN_2 = LayerNormalization()

        LSTM_3 = tf.keras.layers.LSTM(setting.LISM_1_UNTIS, return_sequences=True,activation=tf.keras.activations.linear)
        self.BiLSTM_3 = tf.keras.layers.Bidirectional(LSTM_3, merge_mode="concat")
        self.LN_3 = LayerNormalization()

        self.Att = Attention(units=p_setting.LABEL_SUM, trainable=True)
        self.Prepare_For_Metric = tf.keras.layers.Lambda(self.prepare_for_metric)

    # @tf.function()
    def call(self, inputs, training=None, mask=None):

        output = self.BiLSTM_1(inputs[0],mask=inputs[1])
        output = self.LN_1(output)
        output = self.Att(output, isfinal=False)

        output = self.BiLSTM_2(output,mask=inputs[1])
        output = self.LN_2(output)
        output = self.Att(output, isfinal=False)

        output = self.BiLSTM_3(output,mask=inputs[1])
        output = self.LN_3(output)
        output = self.Att(output, isfinal=True)

        output_for_metric = self.Prepare_For_Metric([output,inputs[1]])

        return [output,output_for_metric]

    def prepare_for_metric(self,inputs):
        "这个是为方便metric的输入而准备的"
        out = tf.math.argmax(inputs[0],output_type=tf.int32,axis=-1)
        mask = tf.cast(inputs[1],dtype=tf.int32)
        return tf.math.multiply(out,mask)

if __name__ == "__main__":
    test_dataset = OutDataset(*setting.LOAD_TEST_PATH)().shuffle(100).batch(setting.BATCH_SIZE,drop_remainder=True).prefetch(3)
    train_dataset = OutDataset(*setting.LOAD_NEW_PATH)().batch(setting.BATCH_SIZE,drop_remainder=True).prefetch(3)
    model = Model()
    op = tf.keras.optimizers.Adam(1e-4)
    save = Save(save_per_epoch=setting.SAVED_EVERY_TIMES,save_directory=setting.MODEL_PATH_SAVE,save_name=setting.MODEL_NAME_SAVE,restore_path=None,save_num=setting.STRAT_NUM)
    model.compile(
        optimizer=op,
        loss=[En_Cross(),Loss_Return_0()],
        metrics=[[],[F1()]]
    )

    model.fit(
        x=train_dataset,
        epochs=setting.EPOCH,
        verbose=1,
        validation_data=test_dataset,
        validation_freq=setting.SAVED_EVERY_TIMES,
        callbacks=[save,tf.keras.callbacks.TensorBoard(setting.LOG_dir)]
    )

