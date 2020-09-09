# @Time: 2020/7/9 15:23
# @Author: R.Jian
# @Note: 主要完成两件事情
# 1 随机生成500个随机的训练集合，作为初始的训练集合。
# 2 随机选择3000条样本，样本选择，其中的五百条。

import random



#HPsetting
ori_sentence_path = "data/ori/train_sentence.txt"
ori_label_path = "data/ori/train_label.txt"

init_sentence_path = "data/init_train_sentence.txt"
init_label_path = "data/init_train_label.txt"

rest_sentence_path = "data/rest_train_sentence.txt"
rest_label_path = "data/rest_train_label.txt"

ready_for_select_train_sentence_path = "data/ready_for_select_train_sentence_path.txt"
ready_for_select_train_label_path = "data/ready_for_select_train_label_path.txt"


def init_dataset():
    random_num = random.sample(range(0,7500),500)
    f_init_sentence = open(init_sentence_path,"w",encoding="utf8")
    f_init_label = open(init_label_path,"w",encoding="utf8")
    f_rest_sentence = open(rest_sentence_path,"w",encoding="utf8")
    f_rest_label = open(rest_label_path,"w",encoding="utf8")
    with open(ori_sentence_path,"r",encoding="utf8") as f_ori_sentence,open(ori_label_path,"r",encoding="utf8") as f_ori_label:
        sentence_label = list(zip(f_ori_sentence.readlines(),f_ori_label.readlines()))

    for i  in range(len(sentence_label)):
        if i in random_num:
            print(sentence_label[i][0],file=f_init_sentence,end="")
            print(sentence_label[i][1],file=f_init_label,end="")
        else:
            print(sentence_label[i][0],file=f_rest_sentence,end="")
            print(sentence_label[i][1],file=f_rest_label,end="")

    f_init_label.close()
    f_init_sentence.close()
    f_rest_sentence.close()
    f_rest_label.close()

def select_data(N):
    '''
    从rest的数据中，选择3000条数据。
    然后添加到ready_for_select_train_***_path.txt中，用来进行样本的选择。
    :return:
    '''
    random_num = random.sample(range(0, N), 500)
    f_ready_for_sentence = open(ready_for_select_train_sentence_path, "w", encoding="utf8")
    f_ready_for_label = open(ready_for_select_train_label_path, "w", encoding="utf8")
    with open(rest_sentence_path, "r", encoding="utf8") as f_ori_sentence, open(rest_label_path, "r",
                                                                               encoding="utf8") as f_ori_label:
        sentence_label = list(zip(f_ori_sentence.readlines(), f_ori_label.readlines()))

    f_rest_sentence = open(rest_sentence_path, "w", encoding="utf8")
    f_rest_label = open(rest_label_path, "w", encoding="utf8")

    for i in range(len(sentence_label)):
        if i in random_num:
            print(sentence_label[i][0], file=f_ready_for_sentence, end="")
            print(sentence_label[i][1], file=f_ready_for_label, end="")
        else:
            print(sentence_label[i][0], file=f_rest_sentence, end="")
            print(sentence_label[i][1], file=f_rest_label, end="")

    f_ready_for_sentence.close()
    f_ready_for_label.close()
    f_rest_sentence.close()
    f_rest_label.close()


# init_dataset()
select_data(5000)