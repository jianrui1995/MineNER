# @Time    : 2020/2/8 1:14
# @Author  : R.Jian
# @NOTE    : 最后的输出，生成对象，然后call调用,

import tensorflow as tf
import preprogram_of_CCKS2019_subtask1.setting as setting
from preprogram_of_CCKS2019_subtask1.pre_sentenceANDsum import SentenceAndSumDataset
from preprogram_of_CCKS2019_subtask1.pre_label import LabelDataset

class OutDataset():
    def __init__(self,sen_path,sum_path,label_path):
        self.senANDsum = SentenceAndSumDataset(sen_path,sum_path)
        self.lab = LabelDataset(label_path)

    def __call__(self, *args, **kwargs):
        return tf.data.Dataset.zip((self.senANDsum(),(self.lab(),self.lab())))

if __name__ == "__main__":
    out = OutDataset(*setting.LOAD_PATH)
    print(out().element_spec)
    for data in out().batch(5).take(3):
        "输出的格式： （（x，mask，sum），y）"
        # mask=tf.cast(data[0][1],dtype=tf.int32)
        print(data)