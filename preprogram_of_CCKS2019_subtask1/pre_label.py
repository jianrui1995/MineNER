# @Time    : 2020/2/8 0:49
# @Author  : R.Jian
# @NOTE    : 

import tensorflow as tf
import preprogram_of_CCKS2019_subtask1.setting as setting

class LabelDataset():
    def __init__(self,label_path):
        self.label_dataset = tf.data.TextLineDataset(label_path)

    def __call__(self, *args, **kwargs):
        label_dataset = self.label_dataset.map(lambda lab:tf.py_function(self.label_into,inp=[lab],Tout=[tf.int32]))
        return label_dataset

    def label_into(self,lab):
        "lab是字符串tensor，需要转换为数字，又map进行调用"
        str_label = lab.numpy().decode()
        list_label = str_label.split(" ")
        list_label = [int(i) for i in list_label]
        for _ in range(len(list_label),setting.MAX_SUM):
            list_label.append(0)
        return [list_label]


if __name__ == "__main__":
    lab = LabelDataset(setting.PRO_TRAIN_LABEL_PATH)
    for data in lab().take(2):
        print(data)