# @Time: 2020/8/17 19:56
# @Author: R.Jian
# @Note: 

import os
import json

def one():
    a = os.walk("ccks_8_data_v2/train")
    f = open("label2id.json",encoding="utf8")
    label2id = json.load(f)
    print(label2id)
    f.close()
    num = 1
    dict_intity = {
        "试验要素":0,
        "性能指标":0,
        "系统组成":0,
        "任务场景":0
    }
    submit = {}
    for data in a:
        for name in data[2]:
            entitys = []
            num = num +1
            path = data[0]+"/"+name
            f = open(path,"r",encoding="GBK")
            dict_info = json.load(f)
            f.close()
            text = dict_info["originalText"].strip("\r\n").replace("\r\n"," ")
            submit[name] = {}
            submit[name]["originalText"] = text
            list_test = list(text)
            for entity in dict_info["entities"]:
                entity_name = list_test[entity["start_pos"]-1:entity["end_pos"]]
                entity["name"] = "".join(entity_name)
                entitys.append(entity)
            submit[name]["entities"] = entitys
    f = open("train_submit.json","w",encoding="utf8")
    json.dump(submit,f,ensure_ascii=False,indent=4)
one()