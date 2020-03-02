# @Time: 2020/2/21 15:04
# @Author: R.Jian
# @Note: MNLP的关键算法

import tensorflow as tf
import sys

sys.path.append(r"D:\pythonProjectList\ProjectOf_NER\MineNER")
sys.path.append(r"/home/tech/myPthonProject/MineNER/")

from BiLSTM_Att.model import Model
from preprogram_of_CCKS2019_subtask1.pre_out import OutDataset
import MNLP_Select.setting as setting

class MNLP():
    def __init__(self,model_path):
        self.bilstm_att = Model()
        self.ckpt = tf.train.Checkpoint(model=self.bilstm_att)
        print(model_path)
        self.ckpt.restore(model_path)
        self.dataset = OutDataset(*setting.LOAD_PATH)

    def __call__(self, *args, **kwargs):
        f = open(setting.PRO_SELETC_FILE_PATH,"w",encoding="utf8")
        for data in self.dataset().batch(1).take(10):
            out = self.bilstm_att(data[0][0],mask=data[0][1])
            out = self.MNLP(out,data[0][2])
            print(str(out),file=f)
        f.close()

    def MNLP(self,input,sum):
        out = input[:,0:sum.numpy()[0],:]
        out = tf.math.reduce_max(out,axis=-1)
        out = tf.math.reduce_sum(out,axis=-1)
        out = tf.math.divide(out,sum.numpy()[0])
        return out.numpy()[0]

if __name__ == "__main__":
    mnlp = MNLP(setting.MODEL_PATH)
    mnlp()