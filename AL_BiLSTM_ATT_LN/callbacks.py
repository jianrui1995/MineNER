# @Time: 2020/7/12 14:31
# @Author: R.Jian
# @Note: 

import tensorflow as tf
from visualdl import LogWriter

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
            self.ckma.save(epoch+1)

class VisualDL(tf.keras.callbacks.Callback):
    def __init__(self,logdir,validation_every_times):
        super(VisualDL,self).__init__()
        self.logdir = logdir
        self.validation_every_times = validation_every_times

    def on_train_begin(self, logs=None):
        self.writer_train = LogWriter(self.logdir+"/train")
        self.writer_val = LogWriter(self.logdir+"/val")

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.writer_train.add_scalar("loss/output_1_loss",logs["output_1_loss"],step=epoch)
        self.writer_train.add_scalar("metric/F1",logs["output_4_F1"],step=epoch)
        self.total_loss = 0.0
        if ((epoch+1) % self.validation_every_times)==0:
            self.writer_val.add_scalar("loss/output_1_loss",logs["val_output_1_loss"],step=epoch)
            self.writer_val.add_scalar("metric/F1", logs["val_output_4_F1"], step=epoch)

class isUseLL(tf.keras.callbacks.Callback):
    def __init__(self,notuseLL):
        super(isUseLL,self).__init__()
        self.notuseLL = notuseLL

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) == self.notuseLL:
            self.model.isUseLL.assign(False)

# 用来进行predict的callbacks
class ForPredict(tf.keras.callbacks.Callback):
    def __init__(self,restore_path):
        super(ForPredict,self).__init__()
        self.restore_path = restore_path

    def set_model(self, model):
        self.model = model
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckpt.restore(self.restore_path)


    def on_predict_begin(self, logs=None):
        self.model.ispredict.assign(True)


    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        print("logs",logs)

    def on_predict_end(self, logs=None):
        pass