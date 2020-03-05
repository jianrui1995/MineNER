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
        self.ckpt.restore(model_path)
        self.dataset = OutDataset(*setting.LOAD_PATH)

    def __call__(self, *args, **kwargs):
        f = open(setting.PRO_SELETC_FILE_PATH,"w",encoding="utf8")
        for data in self.dataset().batch(1):
            out = self.bilstm_att(data[0][0],mask=data[0][1])
            out = self.MNLP(out,data[0][2])
            print(str(out),file=f)
        f.close()
        self.Top_n()

    def MNLP(self,input,sum):
        "计算每个样本概率的平均值"
        out = input[:,0:sum.numpy()[0],:]
        out = tf.math.reduce_max(out,axis=-1)
        out = tf.math.log(out)
        out = tf.math.reduce_mean(out,axis=-1)
        return out.numpy()[0]

    def Top_n(self):
        f = open(setting.PRO_SELETC_FILE_PATH,"r",encoding="utf8")
        list_mean = [[k,float(v.strip())] for k,v in enumerate(f.readlines())]
        f.close()
        list_topN = [[0,1.1] for _ in range(setting.TOP_N+1)]
        list_topN[-1][1] = - sys.float_info.max
        for data in list_mean:
            if data[1]<list_topN[0][1]:
                for i in range(1,len(list_topN)):
                    if data[1]>list_topN[i][1]:
                        list_topN.insert(i,data)
                        list_topN = list_topN[1:]
                        break
        list_topN = list_topN[:-1]
        list_topN.sort(key=lambda x:x[0],reverse=True)

        f = open(setting.PRO_TRAIN_SENTENCE_OLD_PATH,"r",encoding="utf8")
        list_old_sentence = f.readlines()
        f.close()

        f = open(setting.PRO_TRAIN_SUM_OLD_PATH,"r",encoding="utf8")
        list_old_sum = f.readlines()
        f.close()

        f = open(setting.PRO_TRAIN_LABEL_OLD_PATH,"r",encoding="utf8")
        list_old_label = f.readlines()
        f.close()

        f = open(setting.PRO_TRAIN_SENTENCE_NEW_PATH,"r",encoding="utf8")
        list_new_sentence = f.readlines()
        f.close()

        f = open(setting.PRO_TRAIN_SUM_NEW_PATH,"r",encoding="utf8")
        list_new_sum = f.readlines()
        f.close()

        f = open(setting.PRO_TRAIN_LABEL_NEW_PATH,"r",encoding="utf8")
        list_new_label = f.readlines()
        f.close()

        for data in list_topN:
            list_new_sentence.append(list_old_sentence[data[0]])
            list_new_sum.append(list_old_sum[data[0]])
            list_new_label.append(list_old_label[data[0]])
            list_old_sentence.pop(data[0])
            list_old_sum.pop(data[0])
            list_old_label.pop(data[0])

        f = open(setting.PRO_TRAIN_SENTENCE_OLD_PATH, "w", encoding="utf8")
        print("".join(list_old_sentence),file=f,end="")
        f.close()

        f = open(setting.PRO_TRAIN_SUM_OLD_PATH, "w", encoding="utf8")
        print("".join(list_old_sum),file=f,end="")
        f.close()

        f = open(setting.PRO_TRAIN_LABEL_OLD_PATH, "w", encoding="utf8")
        print("".join(list_old_label),file=f,end="")
        f.close()

        f = open(setting.PRO_TRAIN_SENTENCE_NEW_PATH, "w", encoding="utf8")
        print("".join(list_new_sentence),file=f,end="")
        f.close()

        f = open(setting.PRO_TRAIN_SUM_NEW_PATH, "w", encoding="utf8")
        print("".join(list_new_sum),file=f,end="")
        f.close()

        f = open(setting.PRO_TRAIN_LABEL_NEW_PATH, "w", encoding="utf8")
        print("".join(list_new_label),file=f,end="")
        f.close()


if __name__ == "__main__":
    mnlp = MNLP(setting.MODEL_PATH)
    mnlp()
    # mnlp.Top_n()
