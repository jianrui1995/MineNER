# @Time    : 2020/2/6 0:00
# @Author  : R.Jian
# @NOTE    : 实现对原始文件的处理，包括，分离句子和实体，计算句子中字数量的最大值。

import json
import preprogram_of_CCKS2019_subtask1.setting as setting

def devide(ori_path,sentence_path,label_path,sum_path):
    "文件的分离"
    f_ori = open(ori_path,"r",encoding="utf-8-sig")
    f_sentence = open(sentence_path,"w",encoding="utf8")
    f_label = open(label_path,"w",encoding="utf8")
    f_sum = open(sum_path,"w",encoding="utf8")
    data = f_ori.readline()
    while data:
        str_sentences,list_labels,sum = devide_into(data.strip())
        list_sentences = list(str_sentences)
        "句号的下标列表"
        list_period = [i for i,x in enumerate(list_sentences) if x=="。"]
        str_sentence = "".join(list_sentences[:list_period[0]+1])
        list_label = list_labels[:list_period[0]+1]
        sum = len(list_label)
        "写到文件中"
        print(str_sentence,file=f_sentence)
        list_label = [str(i) for i in list_label]
        print(" ".join(list_label),file=f_label)
        print(str(sum),file=f_sum)
        for index in range(1,len(list_period)):
            str_sentence = "".join(list_sentences[list_period[index-1]+1:list_period[index] + 1])
            list_label = list_labels[list_period[index-1]+1:list_period[index] + 1]
            sum = len(list_label)
            "写到文件中"
            print(str_sentence,file=f_sentence)
            list_label = [str(i) for i in list_label]
            print(" ".join(list_label),file=f_label)
            print(str(sum),file=f_sum)
        data = f_ori.readline()
    f_ori.close()
    f_sentence.close()
    f_label.close()
    f_sum.close()

def devide_into(data):
    "def devide 内部调用的函数"
    dict_data = json.loads(data)
    sum = len(dict_data["originalText"])
    list_labels = [setting.TAG2LABEL["o"] for _ in range(sum)]
    for entiey in dict_data["entities"]:
        start_pos = entiey["start_pos"]
        end_pos = entiey["end_pos"]
        name = entiey["label_type"]
        tag = setting.NAME2TAG[name]
        'tag转变为label'
        list_label = tag2label_BIO(tag,start_pos,end_pos)
        '在list_labels中，用list_label进行替换'
        list_labels = list_labels[:start_pos]+list_label+list_labels[end_pos:]
    return dict_data["originalText"],list_labels,sum

def tag2label_BIO(tag,start,end):
    "将实体转变为BIO格式对应的数字代码的列表"
    "BIO-tag列表"
    list_tags = []
    list_tags.append("B_"+tag)
    for _ in range(1,end-start):
        list_tags.append("I_"+tag)
    list_labels = []
    for data in list_tags:
        list_labels.append(setting.TAG2LABEL[data])
    return list_labels

if __name__ == "__main__":
    devide(setting.ORI_TRATIN_PATH,setting.PRO_TRAIN_SENTENCE_PATH,setting.PRO_TRAIN_LABEL_PATH,setting.PRO_TRAIN_SUM_PATH)
    devide(setting.ORI_TEST_PATH,setting.PRO_TEST_SENTENCE_PATH,setting.PRO_TEST_LABEL_PATH,setting.PRO_TEST_SUM_PATH)