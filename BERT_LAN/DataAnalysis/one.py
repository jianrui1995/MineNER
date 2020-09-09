# @Time: 2020/7/7 21:22
# @Author: R.Jian
# @Note: 把json文件变成文本和label文件。
# 返回：两个文件，train_label.txt和train_sentence.txt的文件。
# 分别将其中的句子摘取出来和用BIOES进行打标。 把\r\n替换成空格。
# 然后将空格转换成[space]

import os
import json

def one():
    a = os.walk("ccks_8_data_v2/train")
    f = open("ccks_8_data_v2/label2id.json",encoding="utf8")
    label2id = json.load(f)
    print(label2id)
    f.close()
    f1 = open("one_train_sentence.txt","w",encoding="utf8")
    f2 = open("one_train_label.txt","w",encoding="utf8")
    num = 1
    for data in a:
        for name in data[2]:
            print(num,name)
            num = num +1
            path = data[0]+"/"+name
            f = open(path,"r",encoding="GBK")
            dict_info = json.load(f)
            f.close()
            text = dict_info["originalText"].strip("\r\n").replace("\r\n"," ")
            print(text.replace(" "," [SPACE] "),file=f1)
            list_label = ["0" for _ in range(len(list(text)))]
            for entity in dict_info["entities"]:
                if entity["end_pos"]-entity["start_pos"]+1>2:
                    label = ["B_"+entity["label_type"]]+["I_"+entity["label_type"] for _ in range(entity["end_pos"]-entity["start_pos"]-2+1)]+["E_"+entity["label_type"]]
                    label = [str(label2id[i]) for i in label]
                if entity["end_pos"]-entity["start_pos"]+1==2:
                    label = ["B_" + entity["label_type"]] + ["E_" + entity["label_type"]]
                    label = [str(label2id[i]) for i in label]
                if  entity["end_pos"]-entity["start_pos"]+1==1:
                    label = ["S_" + entity["label_type"]]
                    label = [str(label2id[i]) for i in label]
                list_label[entity["start_pos"]-1:entity["end_pos"]] = label
            print(" ".join(list_label),file=f2)

    f1.close()
    f2.close()

one()