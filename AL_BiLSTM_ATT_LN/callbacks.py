# @Time: 2020/5/11 22:28
# @Author: R.Jian
# @Note: 召回类

import tensorflow as tf
from visualdl import LogWriter
import BiLSTM_Att.setting_model_LN as setting

class Save(tf.keras.callbacks.Callback):
    "执行顺序-init-set_model-set_params"
    def __init__(self,save_per_epoch,save_directory,save_name,restore_path=None):
        super(Save,self).__init__()
        self.save_directory = save_directory
        self.save_name = save_name
        self.save_per_epoch = save_per_epoch
        self.restore_path = restore_path


    def set_params(self, params):
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckma = tf.train.CheckpointManager(self.ckpt,self.save_directory,100,checkpoint_name=self.save_name)

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        if self.restore_path:
            self.ckpt.restore(self.restore_path)

    def on_epoch_end(self, epoch, logs=None):
        if not ((epoch+1) % self.save_per_epoch):
            self.ckma.save(epoch)

class VisualDL(tf.keras.callbacks.Callback):
    def __init__(self,logdir,validation_every_times):
        super(VisualDL,self).__init__()
        self.logdir = logdir
        self.total_loss = 0.0
        self.validation_every_times = validation_every_times

    def on_train_begin(self, logs=None):
        self.writer_train = LogWriter(self.logdir+"/train")
        self.writer_test = LogWriter(self.logdir+"/test")

    def on_batch_end(self, batch, logs=None):
        self.total_loss = self.total_loss + logs["loss"]

    def on_epoch_end(self, epoch, logs=None):
        self.writer_train.add_scalar("loss/loss",self.total_loss,step=epoch)
        self.writer_train.add_scalar("metric/F1",logs["output_2_F1"],step=epoch)
        self.total_loss = 0.0
        if ((epoch+1) % self.validation_every_times)==0:
            self.writer_test.add_scalar("metric/F1",logs["val_output_2_F1"],step=epoch)

