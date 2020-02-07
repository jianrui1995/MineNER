# @Time    : 2020/2/8 1:14
# @Author  : R.Jian
# @NOTE    : 

import tensorflow as tf
import preprogram_of_CCKS2019_subtask1.setting as setting
from preprogram_of_CCKS2019_subtask1.pre_sentenceANDsum import SentenceAndSumDataset
from preprogram_of_CCKS2019_subtask1.pre_label import LabelDataset

class OutDatast():
    def __init__(self,sen_path,sum_path,label_path):
        self.senANDsum = SentenceAndSumDataset(sen_path,sum_path)
        self.lab = LabelDataset(label_path)

    def __call__(self, *args, **kwargs):
        return tf.data.Dataset.zip((self.senANDsum(),self.lab()))

if __name__ == "__main__":
    out = OutDatast(setting.PRO_TRAIN_SENTENCE_PATH,setting.PRO_TRAIN_SUM_PATH,setting.PRO_TRAIN_LABEL_PATH)
    for data in out().batch(2).take(2):
        print(data)