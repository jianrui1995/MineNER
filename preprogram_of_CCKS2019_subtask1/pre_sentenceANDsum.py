# @Time    : 2020/2/7 21:44
# @Author  : R.Jian
# @NOTE    : 

import tensorflow as tf
import gensim
import json
import numpy as np
import preprogram_of_CCKS2019_subtask1.setting as setting

class SentenceAndSumDataset():
    def __init__(self,sentence_path,sum_path):
        sentence_dataset = tf.data.TextLineDataset(sentence_path)
        sum_dataset = tf.data.TextLineDataset(sum_path)
        # 合并sentence和sum的数据集
        self.sentenceANDsum_dataset = tf.data.Dataset.zip((sentence_dataset,sum_dataset))
        "修改字向量的载入方式，修改这里"
        self.model =gensim.models.KeyedVectors.load_word2vec_format(setting.GLOVE_PATH,binary=False)
        setting.MAX_SUM = self.calculate_max_sum(sum_path)

    # @tf.function
    def __call__(self, *args, **kwargs):
        sentenceANDsum_dataset = self.sentenceANDsum_dataset.map(lambda sen,sum: tf.py_function(self.sentence2wordvec,inp=[sen,sum],Tout=[tf.float32,tf.bool,tf.int32]))
        return sentenceANDsum_dataset

    def sentence2wordvec(self,sen,sum):
        "将句子转换为向量，由map方法调用"
        str_sentence = sen.numpy().decode()
        str_sum = sum.numpy().decode()
        int_sum = int(str_sum)
        list_sentence = list(str_sentence)
        unknown = [0 for _ in range(setting.WORDVEC_NUM)]
        vec = []
        "向量转换过程"
        for char in list_sentence:
            vec.append(self.word2vec_model(char))
        "添加多余的部分，保持数据格式一致"
        for _ in range(int_sum,setting.MAX_SUM):
            vec.append(unknown)
        "添加距离向量的过程"
        f = open(setting.LOC_VEC_PATH,"r",encoding="utf8")
        dict_loc = json.load(f)
        f.close()
        for i in range(setting.MAX_SUM):
            vec[i] = np.append(vec[i],np.array(dict_loc[str(i)]))
        "制作mask"
        sum = [True for _ in range(int_sum)]+[False for _ in range(int_sum,setting.MAX_SUM)]
        return [vec,sum,int_sum]

    def word2vec_model(self,char):
        "使用word2vec模型转换字向量"
        unknown = [0 for _ in range(setting.WORDVEC_NUM)]
        try:
            return self.model[char]
        except BaseException:
            return self.model["<unk>"]

    def calculate_max_sum(self,sum_path):
        "计算句子的最大长度"
        f = open(sum_path,"r",encoding="utf8")
        str_sum = f.readline()
        max = 0
        while str_sum:
            int_sum = int(str_sum)
            if max < int_sum:
                max = int_sum
            str_sum = f.readline()
        f.close()
        return max+1



if __name__ == "__main__":
    sen = SentenceAndSumDataset(setting.PRO_TRAIN_SENTENCE_PATH,setting.PRO_TRAIN_SUM_PATH)
    for data in sen().batch(1):
        print(data[2].numpy()[0])