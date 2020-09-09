# @Time: 2020/8/11 22:05
# @Author: R.Jian
# @Note: 生成验证文件

import json
f_label = open("val_result-560.txt", "r", encoding="utf8")
f_word = open("val_word.txt","r",encoding="utf8")
list_type = ["试验要素","性能指标","系统组成","任务场景",]
dict_result = {}
b = 1
for la,wo in zip(f_label.readlines(),f_word.readlines()):
    list_la =[int(i) for i in la.strip().split(" ")]
    list_wo = wo.strip().split(' ')

    start_pos = 0
    end_pos = 0
    type = None
    entities = []
    name = ""

    for i in range(len(list_wo)):
        num = list_la[i]
        if list_la[i] ==0 :
            start_pos += (1 if list_wo[i] in ["[SPACE]","[UNK]"] else len(list(list_wo[i])))
            end_pos += (1 if list_wo[i] in ["[SPACE]","[UNK]"] else len(list(list_wo[i])))
        elif 13>list_la[i]>0:
            if list_la[i]%3 == 1:
                type = list_type[list_la[i]//3]
                start_pos += 1
                end_pos += 1 if list_wo[i] in ["[SPACE]","[UNK]"] else len(list(list_wo[i]))
                name =name+ list_wo[i]

            elif list_la[i]%3 == 2:
                end_pos += 1 if list_wo[i] in ["[SPACE]", "[UNK]"] else len(list(list_wo[i]))
                if type:#有值
                    if type != list_type[list_la[i]//3] :#type和type不一样
                        type =  None
                        start_pos = end_pos
                        name=""
                    else:
                        name=name+list_wo[i]
                else: #无值
                    start_pos = end_pos
                    name=""

            elif list_la[i]%3 == 0:
                end_pos += 1 if list_wo[i] in ["[SPACE]", "[UNK]"] else len(list(list_wo[i]))
                if type: #有值
                    if type == list_type[(list_la[i]-1)//3]: #如果两个type一样
                        name = name+list_wo[i]
                        dict_entity = {
                            "label_type": type,
                            "overlap": 0,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "name":name
                        }
                        entities.append(dict_entity)
                        type=None
                        start_pos = end_pos
                        name=""
                    else:   #如果两个type不一样
                        start_pos = end_pos
                        name = ""
                else: #无值
                    start_pos = end_pos
                    name = ""
        elif 12<list_la[i]<17:
            type = list_type[list_la[i]-13]
            start_pos += 1
            end_pos += 1 if list_wo[i] in ["[SPACE]","[UNK]"] else len(list(list_wo[i]))
            name=list_wo[i]
            dict_entity = {
                "label_type": type,
                "overlap": 0,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "name": name
            }
            entities.append(dict_entity)
            type = None
            start_pos = end_pos
            name=""
    dict_result["validate_V2_"+str(b)+".json"] = entities
    b += 1

f_label.close()
f_word.close()
f_submit = open("val_submit-560.json", "w", encoding="utf8")
json.dump(dict_result,f_submit,ensure_ascii=False,indent=4)