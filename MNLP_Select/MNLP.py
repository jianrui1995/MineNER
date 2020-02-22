# @Time: 2020/2/21 15:04
# @Author: R.Jian
# @Note: MNLP的关键算法

import tensorflow as tf
from BiLSTM_Att.model import Model


class MNLP():
    def __init__(self,model_path):
        bilstm_att = Model()
        ckpt = tf.train.Checkpoint(model=bilstm_att)
        ckpt.restore(model_path)

