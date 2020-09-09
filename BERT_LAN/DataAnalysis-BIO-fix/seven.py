# @Time: 2020/8/19 23:57
# @Author: R.Jian
# @Note: 验证集合去掉子串 保证每个实体只出现一次。

import json

def issubsring(stra,strb):
    print(stra,strb,sep=" ")
    list_stra = list(stra)
    list_strb = list(strb)
    i = 0 #a
    l = 0 #b
    while i < len(list_stra):
        if list_stra[i] == list_strb[l]:
            l+=1
            i+=1
        else:
            i+=1
        if l==len(list_strb)-1:
            return True

    return False

f= open("val_submit.json","r",encoding="utf8")
dict_submit = json.load(f)
f.close()
new_dict_submit = {}
for k,v in  dict_submit.items():
    entities = []
    i = 0
    l = len(v)
    while i < l:
        if v[i]["label_type"]=="试验要素":
            b = i+1
            while b < l:
                if issubsring(v[i]["name"],v[b]["name"]):
                    v.pop(b)
                    l-=1
                else:
                    b+=1
        i+=1
    for entity in v:
        entities.append({
            "label_type": entity["label_type"],
            "overlap": 0,
            "start_pos": entity["start_pos"],
            "end_pos": entity["end_pos"]
            # "name":entity["name"]
        })
    new_dict_submit[k]=entities
f=open("val_fix_submit.json","w",encoding="utf8")
json.dump(new_dict_submit,f,ensure_ascii=False,indent=4)
f.close()