# @Time: 2020/7/7 21:22
# @Author: R.Jian
# @Note: 把json文件变成文本和label文件。
# 返回：两个文件，train_label.txt和train_sentence.txt的文件。
# 分别将其中的句子摘取出来和用BIOES进行打标。 把\r\n替换成空格。
# 然后将空格转换成[space]
# 子串生成和索引


import os
import json
def zero(str_b,start_pos,end_pos,name):
    list_str_b = list(str_b)
    list_str_a = list_str_b[start_pos - 1:end_pos]

    # 0开始的索引
    b_start_point = end_pos
    b_end_point = end_pos
    entity = []

    while b_start_point < len(list_str_b):
        # b中在a的索引集合
        indexs = []
        nnn = list_str_b[b_start_point]
        for i in range(len(list_str_a)):
            if list_str_b[b_start_point] == list_str_a[i]:
                indexs.append(i)
        if len(indexs) > 0:  # indexs 有值
            en = {
                "label_type":name,
                "overlap": 0,
                "start_pos": 0,
                "end_pos": 1,
                "name": "",
                "len": 0
            }
            for index in indexs:
                nn = list_str_b[b_start_point]
                b_end_point = b_start_point + 1
                a_start_point = index
                a_end_point = index + 1
                while (a_end_point < len(list_str_a)) and (b_end_point<len(list_str_b)):
                    if list_str_a[a_end_point] == list_str_b[b_end_point]:
                        a_end_point += 1
                        b_end_point += 1
                    else:
                        a_end_point += 1
                if b_end_point - b_start_point > 2:
                    if a_end_point - a_start_point > en["len"]:
                        en = {
                            "label_type": name,
                            "overlap": 0,
                            "start_pos": b_start_point+1,
                            "end_pos": b_end_point,
                            "name": "".join(list_str_b[b_start_point:b_end_point]),
                            "len": b_end_point - b_start_point
                        }
            if en["len"]:
                entity.append(en)
            b_start_point = b_start_point + (en["end_pos"] - en["start_pos"])+1
        else:  # indexs 无值
            b_start_point += 1
    return entity

def one():
    f = open("label2id.json",encoding="utf8")
    label2id = json.load(f)
    print(label2id)
    f.close()

    f1 = open("one_train_sentence.txt","w",encoding="utf8")
    f2 = open("one_train_label.txt","w",encoding="utf8")
    f = open("zero_train_submit.json","r",encoding="utf8")
    dict_ori = json.load(f)
    f.close()
    dict = {}
    num =1
    for k,v in dict_ori.items():
        print(num,k)
        num +=1
        text = v["originalText"].strip("\r\n").replace("\r\n"," ")
        print(text.replace(" "," [SPACE] "),file=f1)
        # 子串
        new_entity = []
        list_label = [0 for _ in range(len(list(text)))]
        _list_label_for_label = [0 for _ in range(len(list(text)))]
        for entity in v["entities"]:
            #形成label for label
            b = [1] + [1 for _ in range(entity["end_pos"] - entity["start_pos"])]
            _list_label_for_label[entity["start_pos"] - 1:entity["end_pos"]] = b

        for entity in v["entities"]:
            # 只查找“试验要素”的子串
            if (entity["label_type"]=="试验要素") and (entity["end_pos"]-entity["start_pos"]>5):
                a = zero(text,entity["start_pos"],entity["end_pos"],entity["label_type"])
                d = []
                for en in a:
                    slice_en = _list_label_for_label[en["start_pos"]-1:en["end_pos"]]
                    b = [1] + [1 for _ in range(en["end_pos"] - en["start_pos"])]
                    if sum(slice_en) != sum(b):
                        if sum(slice_en) == 0 :
                            en["new"] = " ".join([str(i) for i in b])
                            en["old"] = " ".join([str(i) for i in slice_en])
                            d.append(en)
                new_entity = new_entity + d
            # 查找短实体的完全匹配实体
            elif (entity["label_type"] in ["试验要素","系统组成","性能指标","任务场景"]) and (entity["end_pos"]-entity["start_pos"]<6):
                d = []
                name = entity["name"]
                len1 = entity["end_pos"]-entity["start_pos"]+1
                i = entity["end_pos"]
                while i<len(text):
                    s = text.find(name,i)
                    if s != -1:
                        slice_en = _list_label_for_label[s+1-1:s+len1]
                        b = [1] + [1 for _ in range(s+len1-s-1)]
                        if sum(slice_en) != sum(b):
                            if sum(slice_en) == 0:
                                en = {
                                    "label_type":entity["label_type"],
                                    "overlap":0,
                                    "start_pos":s+1,
                                    "end_pos":s+len1,
                                    "name":name,
                                    "label":222,
                                    "new":" ".join([str(i) for i in b]),
                                    "old":" ".join([str(i) for i in slice_en])
                                }
                                d.append(en)
                        i = s+len1
                    else:
                        break
                new_entity = new_entity + d
        # 交叉对比,执行两遍
        new_entity = sorted(new_entity,key= lambda x:x["start_pos"])
        # 第一遍
        new_new_entity = []
        i = 0
        while i < len(new_entity):
            if i < len(new_entity)-1:
                if new_entity[i]["end_pos"]>new_entity[i+1]["start_pos"]:
                    if new_entity[i]["end_pos"]-new_entity[i]["start_pos"] >= new_entity[i+1]["end_pos"]-new_entity[i+1]["start_pos"]:
                        new_new_entity.append(new_entity[i])
                    else:
                        new_new_entity.append(new_entity[i+1])
                    i+=2
                else:
                    new_new_entity.append(new_entity[i])
                    i+=1
            else:
                new_new_entity.append(new_entity[i])
                i+=1
        new_entity = new_new_entity
        # 第二遍
        new_new_entity = []
        i = 0
        while i < len(new_entity):
            if i < len(new_entity) - 1:
                if new_entity[i]["end_pos"] > new_entity[i + 1]["start_pos"]:
                    if new_entity[i]["end_pos"] - new_entity[i]["start_pos"] >= new_entity[i + 1]["end_pos"] - new_entity[i + 1]["start_pos"]:
                        new_new_entity.append(new_entity[i])
                    else:
                        new_new_entity.append(new_entity[i + 1])
                    i += 2
                else:
                    new_new_entity.append(new_entity[i])
                    i += 1
            else:
                new_new_entity.append(new_entity[i])
                i += 1
        new_entity = new_new_entity


        for entity in new_entity:
            # 对子串实体进行处理
            label = ["B_" + entity["label_type"]] + ["I_" + entity["label_type"] for _ in range(entity["end_pos"] - entity["start_pos"])]
            label = [1 for i in label]
            list_label[entity["start_pos"] - 1:entity["end_pos"]] = [i+j for i,j in zip(list_label[entity["start_pos"] - 1:entity["end_pos"]],label)]

        # for entity in v["entities"]:
        #     # 对真实实体进行处理，
        #     label = ["B_" + entity["label_type"]] + ["I_" + entity["label_type"] for _ in range(entity["end_pos"] - entity["start_pos"])]
        #     label = [1 for i in label]
        #     list_label[entity["start_pos"] - 1:entity["end_pos"]] = [i + j for i, j in zip(list_label[entity["start_pos"] - 1:entity["end_pos"]], label)]

        print(" ".join([str(i) for i in list_label]),file=f2)
        dict[k]={
            "originalText":v["originalText"],
            "entities":v["entities"],
            "new_entities":new_entity
        }
    f1.close()
    f2.close()
    f = open("one_train_submit.json","w",encoding="utf8")
    json.dump(dict,f,ensure_ascii=False,indent=4)
    f.close()
one()