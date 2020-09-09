# @Time: 2020/7/8 10:37
# @Author: R.Jian
# @Note: 生成two_train_label.txt 和 two_train_sentence.txt的两个文件。
# 这两个文件已经是被tonizer过的，只保留了其中的id和对应的分类。

import transformers


# HPsetting
train_sentence_path = "one_train_sentence_div.txt.txt"
train_label_path = "one_train_label_div.txt.txt"
two_train_sentence_path = "two_train_sentence.txt"
two_train_label_path = "two_train_label.txt"
two_train_wordpiece_path = "two_train_wordpiece.txt"

def two():
    tokenizer = transformers.BertTokenizer("../model/chinese_L-12_H-768_A-12/vocab.txt")
    tokenizer.add_special_tokens({"additional_special_tokens":["[SPACE]","“","”"]})
    vocab_f = open("../model/chinese_L-12_H-768_A-12/vocab.txt","r",encoding="utf8")
    list_vocab = vocab_f.readlines()
    list_vocab = [data.strip() for data in list_vocab]
    dict_vocab = {k:v for k,v in enumerate(list_vocab)}

    f_sentence = open(two_train_sentence_path,"w",encoding="utf8")
    f_label = open(two_train_label_path,"w",encoding="utf8")
    f_wordpiece = open(two_train_wordpiece_path, "w", encoding="utf8")


    with open(train_sentence_path,"r",encoding="utf8") as f1,open(train_label_path,"r",encoding="utf8") as f2:
        test_list_label = []
        for sentence,label in zip(f1.readlines(),f2.readlines()):
            sentence = sentence.strip()
            label = label.strip()
            # 目标所求
            # tokened_text = tokenizer.encode(sentence)[1:-1] #全部是id 摘除[cls]
            tokened_text = tokenizer.encode(sentence)  #全部是id
            tokened_text_str = [str(i) for i in tokened_text]
            print(" ".join(tokened_text_str[1:-1]),file=f_sentence)
            print(len(tokened_text_str[1:-1]),end="",sep=" ")
            tokened_text = tokened_text[1:-1]
            # print(tokened_text)
            tokened_list_text = [dict_vocab[data].lstrip("##") for data in tokened_text]
            print(" ".join(tokened_list_text),file=f_wordpiece)
            print(len(tokened_list_text), end="",sep=" ")
            # print(A)
            # print(tokened_list_text)
            list_label = label.split(" ")
            #目标所求。
            list_label_final = []

            for i in range(len(tokened_list_text)):

                num = len(list(tokened_list_text[i]))
                if tokened_list_text[i] in ["[SPACE]","[UNK]"]:
                    # print(1)
                    num =1
                # 这里处理合并问题
                maybe = list_label[0:num]
                if num != 1:
                    maybe_a = [int(i)%2 for i in maybe]
                    if maybe_a[0]==1 and maybe_a[-1]==0:
                        a = maybe[0]
                    elif maybe_a[0]==0 and maybe_a[-1]==0:
                        a = maybe[0]
                    else:
                        a = maybe[0]
                else:
                    a = maybe[0]

                list_label_final.append(a)
                list_label = list_label[num:]
            print(" ".join(list_label_final),file=f_label)
            print(len(list_label_final), end="\n",sep=" ")
            # print(list_label_final)
            # print(list_label_final)

            # test_list_label = test_list_label + list_label_final
        # uniquid = []
        # for data in test_list_label:
        #     if isinstance(data,list):
        #         uniquid.append(data)
        # print(1111)
        # for data in uniquid:
        #     print(data)


two()


