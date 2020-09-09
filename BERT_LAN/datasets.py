# @Time: 2020/7/22 14:53
# @Author: R.Jian
# @Note: 

import tensorflow as tf
import  BERT_LAN.HPsetting as HP

class Sentence():
    def __init__(self,sentence_path):
        self.sentence = tf.data.TextLineDataset(sentence_path)

    def __call__(self, *args, **kwargs):
        out = self.sentence.map(self.convert)
        return out

    def convert(self,sentence):
        def _convert(sentence):
            str_sentence = sentence.numpy().decode()
            list_sentence = str_sentence.split(" ")
            list_sentence = [int(i) for i in list_sentence]
            list_mask = [1 for _ in range(len(list_sentence))]
            return [list_sentence,list_mask]
        out = tf.py_function(_convert,inp=[sentence],Tout=[tf.int32,tf.int32])
        sen = tf.convert_to_tensor(out[0],dtype=tf.int32)
        sen.set_shape([None])
        mask = tf.convert_to_tensor(out[1],dtype=tf.int32)
        mask.set_shape([None])
        return [sen,mask]

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
            list_label = [int(i) for i in list_label]
            list_labels = []
            for i in list_label:
                label_zeros = [0 for _ in range(HP.LABEL_NUM)]
                label_zeros[i] = 1
                list_labels.append(label_zeros)
            return [list_labels]

        # 特别注意: 这里返回的是[list_labels]这个列表
        list_label = tf.py_function(_convert, inp=[label], Tout=[tf.int32])
        # a.set_shape((None))
        list_label = tf.convert_to_tensor(list_label[0],tf.int32)
        list_label.set_shape((None,HP.LABEL_NUM))
        return list_label

class Out():
    def __init__(self,sentence_path,label_path):
        self.sentence = Sentence(sentence_path)
        self.label = Label(label_path)

    def __call__(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return:[(x,mask),y] x: [timestep],tf.int32; mask:[timestep],tf.int32; y:[timestep,labelnum],tf.int32
        '''
        x = self.sentence().map(lambda x,y:x)
        mask = self.sentence().map(lambda x,y:y)
        return tf.data.Dataset.zip(((x,mask),self.label()))


if __name__ == "__main__":

    # 测试sentence
    # i = 1
    # dataset = Sentence(
    #     sentence_path="DataAnalysis/two_train_sentence.txt",
    # )
    # for data in dataset().padded_batch(
    #     batch_size=1,
    #     padded_shapes=([None],[None]),
    #     padding_values=(0,0)
    # ):
    #     print(i)
    #     print(data)
    #     i = i + 1

    # 测试Label
    dataset = Label("DataAnalysis/two_train_label.txt")
    for data in dataset().padded_batch(
        batch_size=8,
        padded_shapes=([None,HP.LABEL_NUM]),
        padding_values=(0)
    ).take(2):
        tf.print(tf.shape(data), summarize=200)
        tf.print(data,summarize=200)

    # 测试Out
    # dataset = Out("DataAnalysis/two_train_sentence.txt","DataAnalysis/two_train_label.txt")
    # for data in dataset().padded_batch(
    #     batch_size=3,
    #     # padded_shapes=(([None],[None]),[None,None]),
    #     padding_values=((0,0),0)
    # ).take(1):
    #     print(data)