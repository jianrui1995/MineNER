# @Time: 2020/7/10 15:21
# @Author: R.Jian
# @Note: 数据分析的预处理程序。生成json文件

import json

# HPsetting
N =3
two_train_label = "two_train_label.txt"
two_train_sentence = "two_train_sentence.txt"
three_train_info = "three_train_info_"+str(N)+".json"

def three(N):
    result = {}
    with open(two_train_sentence,"r",encoding="utf8") as f1,open(two_train_label,"r",encoding="utf8") as f2:
        sentence_label = zip(f1.readlines(),f2.readlines())
        for s,l in sentence_label:
            sentence = s.strip().split(" ")
            label = l.strip().split(" ")
            add_sentence = [10 for i in range(N//2) ] + [int(i) for i in sentence]+ [10 for i in range(N//2)]
            sentence = [int(i) for i in sentence]
            label = [int(i) for i in label]
            for i in range(len(sentence)):
                result.setdefault(sentence[i],{}).setdefault(label[i],[]).append(add_sentence[i:i+N])
    f = open(three_train_info,"w",encoding="utf8")
    json.dump(result,f,ensure_ascii=False)

three(N)