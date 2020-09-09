# @Time: 2020/8/17 22:40
# @Author: R.Jian
# @Note: 从train_submit.json而来，进行第一次修正，修正在标准标注过程中，存在的实体莫名断开的情况。
#  处理因为问号和空格引起的实体断开

import json

f = open("train_submit.json","r",encoding="utf8")
train_submit = json.load(f)
f.close()
new_train_submit = {}
for k,v in train_submit.items():
    a=sorted(v["entities"],key=lambda x:x["start_pos"])
    new = []
    temp  = a[0]
    for i in  range(1,len(a)):
        if (a[i]["start_pos"]-temp["end_pos"] == 1) and (a[i]["label_type"]==temp["label_type"]):
            temp = {
                'label_type': a[i]["label_type"],
                'overlap': 0,
                'start_pos': temp["start_pos"],
                'end_pos':a[i]["end_pos"],
                "name":temp["name"]+a[i]["name"]
            }
            if  i==len(a)-1:
                new.append(temp)
        else:
            new.append(temp)
            temp = a[i]
            if  i==len(a)-1:
                new.append(temp)

    # 处理因为空格和问号引起的实体断开。
    i = 0
    l = len(new)
    new_1 = []
    while i < l:
        if i == l-1:
            new_1.append(new[i])
            i += 1
        # 只有一个字的间隔的时候
        elif (new[i]["end_pos"] == new[i+1]["start_pos"]-2) and (new[i]["label_type"]==new[i+1]["label_type"]) and (new[i]["label_type"] in ["试验要素","系统组成","性能指标","任务场景"]):
            if (list(v["originalText"])[new[i]["end_pos"]] in [" ","?"]) and ((list(v["originalText"])[new[i+1]["end_pos"]]  not in ["和","?","及"]) and ("".join(list(v["originalText"])[new[i+1]["end_pos"]:new[i+1]["end_pos"]+2]) not in ["以及"])):
                new_1.append(
                    {
                        'label_type': new[i]["label_type"],
                        'overlap': 0,
                        'start_pos': new[i]["start_pos"],
                        'end_pos': new[i+1]["end_pos"],
                        "name": "".join(list(v["originalText"])[new[i]["start_pos"]-1:new[i+1]["end_pos"]])
                    }
                )
                print(k)
                i+=2
            else:
                new_1.append(new[i])
                i+=1

        else:
            new_1.append(new[i])
            i += 1
    dict_ori = {
        "originalText":v["originalText"],
        "entities":new_1
    }

    new_train_submit[k]=dict_ori

dict_intity = {
        "试验要素":0,
        "性能指标":0,
        "系统组成":0,
        "任务场景":0
    }
for k,v in new_train_submit.items():
    for entity in v["entities"]:
        dict_intity[entity["label_type"]] +=1
print(dict_intity)

f=open("zero_train_submit.json","w",encoding="utf8")
json.dump(new_train_submit,f,ensure_ascii=False,indent=4)
f.close()






