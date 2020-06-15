# @Time: 2020/5/11 22:28
# @Author: R.Jian
# @Note: 召回类

import tensorflow as tf

class Save(tf.keras.callbacks.Callback):
    "执行顺序-init-set_model-set_params"
    def __init__(self,save_per_epoch,save_directory,save_name,restore_path=None,save_num=1):
        super(Save,self).__init__()
        self.save_directory = save_directory
        self.save_name = save_name
        self.save_per_epoch = save_per_epoch
        self.restore_path = restore_path
        self.save_num = save_num

    def set_params(self, params,batch_size=4):
        self.params = params
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckma = tf.train.CheckpointManager(self.ckpt,self.save_directory,100,checkpoint_name=self.save_name)

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        if self.restore_path:
            self.ckpt.restore(self.restore_path)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_per_epoch:
            self.ckma.save(self.save_num)
            self.save_num = self.save_num + 1