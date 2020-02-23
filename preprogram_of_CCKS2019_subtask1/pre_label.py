# @Time    : 2020/2/8 0:49
# @Author  : R.Jian
# @NOTE    : 

import tensorflow as tf
import preprogram_of_CCKS2019_subtask1.setting as setting

class LabelDataset():

    def __init__(self,label_path):
        self.label_dataset = tf.data.TextLineDataset(label_path)

    def label_into(self,lab):
        "lab是字符串tensor，需要转换为数字，又map进行调用"
        str_label = lab.numpy().decode()
        list_label = str_label.split(" ")
        list_labels = []
        for i in list_label:
            list_allzeor = [0 for _ in range(setting.LABEL_SUM)]
            list_allzeor[int(i)] = 1
            list_labels.append(list_allzeor)
        for _ in range(len(list_label),setting.MAX_SUM):
            list_labels.append([0 for _ in range(setting.LABEL_SUM)])
        return [list_labels]

    @tf.function
    def __call__(self, *args, **kwargs):
        label_dataset = self.label_dataset.map(lambda lab:tf.py_function(self.label_into,inp=[lab],Tout=[tf.float32]))
        return label_dataset


if __name__ == "__main__":
    lab = LabelDataset(setting.PRO_TRAIN_LABEL_PATH)
    for data in lab().take(2):
        print(data)