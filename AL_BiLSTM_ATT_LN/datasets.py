# @Time: 2020/7/7 15:08
# @Author: R.Jian
# @Note: 构建dataset用于数据的输入
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import tensorflow as tf
import gensim
import json
import numpy as np
import AL_BiLSTM_ATT_LN.HPsetting as setting

class Sentence2Vec():
    def __init__(self,sentence_path,glove_vec_path,loc_vec_path):
        '''

        :param sentence_path: sentence,文本文件的路径
        :param glove_vec_path: 词向量的路径
        :param loc_vec_path: 位置向量的路径
        '''
        self.sentence_dataset = tf.data.TextLineDataset(sentence_path)
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(glove_vec_path,binary=False)
        self.loc_vec_path = loc_vec_path

    def __call__(self, *args, **kwargs):
        '''
        执行sentence到vec的转换过程。
        :param args:
        :param kwargs:
        :return:
        '''
        vec_dataset = self.sentence_dataset.map(self.convert)
        return vec_dataset

    def convert(self,sentence):
        '''
        转换过程
        :param sentence: 为对应的string的tensor
        :return: sentenc对应的vec向量集合
        '''
        def _convert(sentence):
            str_sentence = sentence.numpy().decode()
            list_sentence = list(str_sentence)
            vec = []
            # token转换为vec
            for token in list_sentence:
                vec.append(self.token2vec(token))
            # 添加举例向量
            f = open(self.loc_vec_path,"r",encoding="utf8")
            dict_loc = json.load(f)
            f.close()
            for i in range(len(vec)):
                vec[i] = np.append(vec[i],np.array(dict_loc[str(i)]))
            # 制作mask
            mask = [True for _ in range(len(vec))]

            return [vec,mask]

        # 转换为tensor
        # vec = tf.convert_to_tensor(vec,dtype=tf.float32)
        # mask = tf.convert_to_tensor(mask,dtype=tf.bool)
        vec,mask = tf.py_function(_convert, inp=[sentence], Tout=[tf.float32, tf.bool])
        vec = tf.convert_to_tensor(vec,tf.float32)
        vec.set_shape((None,None))
        mask = tf.convert_to_tensor(mask,tf.bool)
        mask.set_shape((None,))
        return [vec,mask]

    def token2vec(self,token):
        '''
        token转回为vec
        :param token: char
        :return: 返回char对应的向量，如果没有char对应的向量则返回<unk>对应的向量。
        '''
        try:
            return self.word2vec_model[token]
        except BaseException:
            return self.word2vec_model["<unk>"]

class Label():
    def __init__(self,label_path):
        '''

        :param label_path: label文件的路径
        '''
        self.label = tf.data.TextLineDataset(label_path)


    def __call__(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return: 转换的执行过程
        '''
        label = self.label.map(self.convert)
        return label

    def convert(self,label):
        '''
        转换的执行程序
        :param label: string的tensor
        :return:
        '''
        def _convert(label):
            str_label = label.numpy().decode()
            list_label = str_label.split(" ")
            # tf.print(list_label)
            list_label = [int(i) for i in list_label]
            list_labels = []
            for i in list_label:
                label_zeros = [0 for _ in range(setting.LABEL_NUM)]
                label_zeros[i] = 1
                list_labels.append(label_zeros)
            return [list_labels]

        # 特别注意: 这里返回的是[list_labels]这个列表
        list_label = tf.py_function(_convert, inp=[label], Tout=[tf.int32])
        zero = tf.constant([0],dtype=tf.int32)
        # a.set_shape((None))
        list_label = tf.convert_to_tensor(list_label[0],tf.int32)
        list_label.set_shape((None,None))
        return [zero,zero,zero,list_label]

class Out():
    def __init__(self,sentence_path,glove_vec_path,loc_vec_path,label_path):
        '''

        :param sentence_path: sentence,文本文件的路径
        :param glove_vec_path: 词向量的路径
        :param loc_vec_path: 位置向量的路径
        :param label_path: label文件的路径
        '''
        self.vec = Sentence2Vec(sentence_path,glove_vec_path,loc_vec_path)
        self.label = Label(label_path)

    def __call__(self, *args, **kwargs):
        '''
        将dataset组合成需要的格式[(x,mask,y),(0,0,0,y)] y的size为[timesetp,label num]
        :param args:
        :param kwargs:
        :return:
        '''
        x = self.vec().map(lambda x,y:x)
        mask = self.vec().map(lambda x,y:y)
        y = self.label().map(lambda x,y,z,w:w)
        # out = tf.data.Dataset.zip((x,self.label()))
        return tf.data.Dataset.zip(((x,mask,y),self.label()))

class Out_For_Predict():
    def __init__(self,sentence_path,glove_vec_path,loc_vec_path,label_path):
        self.vec = Sentence2Vec(sentence_path,glove_vec_path,loc_vec_path)
        self.label = Label(label_path)

    def __call__(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return:
        '''
        x = self.vec().map(lambda x,y:x)
        mask = self.vec().map(lambda x,y:y)
        y = self.label().map(lambda x, y, z, w: w)
        return tf.data.Dataset.zip( ((x,mask,y),(y)) )



if __name__ == "__main__":

    pass
    # 测试sentence2vec
    # dataset = Sentence2Vec(
    #     sentence_path="data/ori/test_sentence.txt",
    #     glove_vec_path="model/glove/vectors.txt",
    #     loc_vec_path="data/ori/loc_vec.json"
    # )
    # for data in dataset().padded_batch(
    #     batch_size=3,
    #     padded_shapes=([None,None],[None]),
    #     padding_values=(0.0,False)
    # ).take(1):
    #     print(data)

    # 测试Label
    dataset = Label("data/init_train_label.txt")
    i = 1
    for data in dataset().padded_batch(
        batch_size=1,
        padded_shapes=([None],[None],[None],[None,None]),
        padding_values=(0,0,0,0)
    ):
        print(i)
        # print(data)
        i = i +1

    # 测试out
    # dataset = Out(
    #     sentence_path="data/init_train_sentence.txt",
    #     glove_vec_path="model/glove/vectors.txt",
    #     loc_vec_path="data/ori/loc_vec.json",
    #     label_path="data/init_train_label.txt"
    # )
    # i = 1
    # for data in dataset().padded_batch(
    #     batch_size=1,
    #     padded_shapes=(([None,None],[None],[None,None]),([None],[None],[None],[None,None]))
    # ):
    #     print(i)
    #     # print(data)
    #     i = i +1

    # 测试out_for_predict
    # dataset = Out_For_Predict(*setting.LOAD_OLD_PATH)
    # for data in dataset().padded_batch(
    #     batch_size=3,
    #     padded_shapes=(([None,None],[None],[None,None]),([None,None])),
    #     padding_values=((0.0,False,0),(0))
    # ).take(2):
    #     print(data)