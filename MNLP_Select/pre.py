# @Time: 2020/2/21 21:33
# @Author: R.Jian
# @Note: 从训练集集合中，随机生成一百个初始的训练集合

import MNLP_Select.setting as setting
import numpy as np

list_random = np.random.randint(0,7650,100)

f_sentence_NEW = open(setting.PRO_TRAIN_SENTENCE_NEW_PATH,"w",encoding="utf8")
f_sum_NEW = open(setting.PRO_TRAIN_SUM_NEW_PATH,"w",encoding="utf8")
f_label_NEW = open(setting.PRO_TRAIN_LABEL_NEW_PATH,"w",encoding="utf8")

f_sentence_OLD = open(setting.PRO_TRAIN_SENTENCE_OLD_PATH,"w",encoding="utf8")
f_sum_OLD = open(setting.PRO_TRAIN_SUM_OLD_PATH,"w",encoding="utf8")
f_label_OLD = open(setting.PRO_TRAIN_LABEL_OLD_PATH,"w",encoding="utf8")

f_sentence = open(setting.PRO_TRAIN_SENTENCE_PATH,"r",encoding="utf8")
f_sum = open(setting.PRO_TRAIN_SUM_PATH,"r",encoding="utf8")
f_label = open(setting.PRO_TRAIN_LABEL_PATH,"r",encoding="utf8")

list_sentence = f_sentence.readlines()
list_sum = f_sum.readlines()
list_label = f_label.readlines()
for num in range(len(list_sentence)):
    if num in list_random:
        print(list_sentence[num].strip("\n"),file=f_sentence_NEW)
        print(list_sum[num].strip("\n"),file=f_sum_NEW)
        print(list_label[num].strip("\n"),file=f_label_NEW)
    else:
        print(list_sentence[num].strip("\n"),file=f_sentence_OLD)
        print(list_sum[num].strip("\n"),file=f_sum_OLD)
        print(list_label[num].strip("\n"),file=f_label_OLD)

f_label_NEW.close()
f_label_OLD.close()
f_label.close()
f_sum_NEW.close()
f_sum.close()
f_sum_OLD.close()
f_sentence_NEW.close()
f_sentence.close()
f_sentence_OLD.close()
