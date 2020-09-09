# @Time: 2020/7/8 10:37
# @Author: R.Jian
# @Note: 用于验证集的预处理

import json
import transformers

f = open("ccks_8_data_v2/validate_data.json",encoding="utf8")
dict_vali = json.load(f)
f.close()

# token
tokenizer = transformers.BertTokenizer("../model/chinese_L-12_H-768_A-12/vocab.txt")
tokenizer.add_special_tokens({"additional_special_tokens": ["[SPACE]", "“", "”"]})
vocab_f = open("../model/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf8")
list_vocab = vocab_f.readlines()
list_vocab = [data.strip() for data in list_vocab]
dict_vocab = {k: v for k, v in enumerate(list_vocab)}

f_sentence = open("val_token.txt", "w", encoding="utf8")
f_word = open("val_word.txt","w",encoding="utf8")

for i in range(1,101):
    sen = dict_vali["validate_V2_"+str(i)+".json"]
    sen = sen.strip("\r\n").replace("\r\n"," ")
    sen = sen.replace(" "," [SPACE] ")
    list_token = tokenizer.encode(sen)
    list_token_str = [str(i) for i in list_token]
    print(" ".join(list_token_str), file=f_sentence)
    list_token = list_token[1:-1]
    print(list_token)
    tokened_list_text = [dict_vocab[data].lstrip("##") for data in list_token]
    print(" ".join(tokened_list_text), file=f_word)
