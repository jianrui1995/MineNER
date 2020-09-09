# @Time    : 2020/9/2 23:01
# @Author  : R.Jian
# @Note    : 将生成的one_train_submit转换成对应的label文档

import json

f = open("label2id.json", encoding="utf8")
label2id = json.load(f)
print(label2id)
f.close()

f = open("one_train_submit.json","r",encoding="utf8")
f2 = open("one_train_label.txt","w",encoding="utf8")
dict_ori = json.load(f)
for k,v in dict_ori.items():
    list_label = ["0" for _ in range(len(v["originalText"]))]
    for entity in v["new_entities"]:
        label = ["B_" + entity["label_type"]] + ["I_" + entity["label_type"] for _ in range(entity["end_pos"] - entity["start_pos"])]
        label = [str(label2id[i]) for i in label]
        list_label[entity["start_pos"] - 1:entity["end_pos"]] = label
    for entity in v["entities"]:
        # 对真实实体进行处理，
        label = ["B_" + entity["label_type"]] + ["I_" + entity["label_type"] for _ in range(entity["end_pos"] - entity["start_pos"])]
        label = [str(label2id[i]) for i in label]
        list_label[entity["start_pos"] - 1:entity["end_pos"]] = label
    print(" ".join(list_label), file=f2)
f2.close()