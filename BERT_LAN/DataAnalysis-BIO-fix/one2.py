# @Time    : 2020/9/6 14:55
# @Author  : R.Jian
# @Note    : 句子切分

import  json

f = open("one_train_submit.json","r",encoding="utf8")
dict_ori = json.load(f)
f.close()
f = open("one_train_label.txt","r",encoding="utf8")

f1 = open("one_train_label_div.txt","w",encoding="utf8")
f2 = open("one_train_sentence_div.txt","w",encoding="utf8")

def is_have_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False
for (k,v),o in zip(dict_ori.items(),f.readlines()):
    text = v["originalText"]
    list_text = list(text)
    list_text_a = [i for i in list_text]
    entities = v["entities"]+v["new_entities"]
    entities = sorted(entities,key=lambda x:x["end_pos"],reverse=True)
    index = []
    for i in range(len(list_text)):
        if list_text[i]=="?":
            list_text[i] = "?-"+str(i)
        i+=1

    for entity in entities:
        list_text[entity["start_pos"]-1:entity["end_pos"]] = " "
        list_text[entity["start_pos"]-1] = "<"+entity["label_type"]+">"

    i = 0
    while i < len(list_text)-1:
        a = list_text[i]
        if list_text[i].startswith("?-"):
            if list_text[i-1]==list_text[i+1] and list_text[i-1] in ["<任务场景>","<性能指标>","<系统组成>","<试验要素>"]:
                list_text[i] = "?"
                i+=1
                continue
            else :
                l = 1
                while l<7:
                    if l == 1:
                        if (not is_have_chinese(list_text[i+l])) and (not is_have_chinese(list_text[i-l])):
                            list_text[i] = "?"
                            break
                    if i-l>=0 :
                        if list_text[i-l] in ["和","并","与"]:
                            list_text[i] = "?"
                            break
                        if list_text[i-l] in ["<任务场景>","<性能指标>","<系统组成>","<试验要素>"]:
                            break
                    if i+l<len(list_text):
                        if list_text[i+l] in ["和","并","与"] :
                            list_text[i] = "?"
                            break
                        if list_text[i+l] in ["<任务场景>","<性能指标>","<系统组成>","<试验要素>"]:
                            break
                    l+=1
                l = i-1
                while l>=0:
                    if list_text[l] in ["?"]:
                        list_text[i] = "?"
                        break
                    if list_text[l] in ["<任务场景>","<性能指标>","<系统组成>","<试验要素>"]:
                        break
                    if l == 0:
                        list_text[i] = "?"
                    l-=1
        i+=1
    for i in list_text:
        if i.startswith("?-"):
            index.append(int(i.split("-")[1]))
    list_label = o.strip().split(" ")
    if index:
        if index[-1] == len(list_label)-1:
            print(1)
            index = [-1]+index
        else:
            index = [-1]+index+[len(list_label)]
    i = 1
    new_index = []
    while i <len(index):
        new_index.append([index[i-1]+1,index[i]+1])
        i+=1
    for s,e in new_index:
        print("".join(list_text_a[s:e]),file=f2)
        print(" ".join(list_label[s:e]),file=f1)


    # print(list_label)
    # print(list_text_a)

