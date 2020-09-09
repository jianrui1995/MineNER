# @Time: 2020/7/26 19:32
# @Author: R.Jian
# @Note: 

# HPsetting
score_path = "data/score.txt"

ori_sentence_path = "data/ori/train_sentence.txt"
ori_label_path = "data/ori/train_label.txt"

init_sentence_path = "data/init_train_sentence.txt"
init_label_path = "data/init_train_label.txt"

rest_sentence_path = "data/rest_train_sentence.txt"
rest_label_path = "data/rest_train_label.txt"

ready_for_select_train_sentence_path = "data/ready_for_select_train_sentence_path.txt"
ready_for_select_train_label_path = "data/ready_for_select_train_label_path.txt"
def select():
    with open(score_path,"r",encoding="utf8") as fs:
        list_score = fs.readlines()
        list_score = [(k,float(v.strip())) for k,v in enumerate(list_score)]
        list_score = sorted(list_score,key=lambda x:x[1],reverse=True)
        select_score = list_score[0:500]
        list_k  = [i[0] for i in select_score]

    f_i_s = open(init_sentence_path,"a+",encoding="utf8")
    f_i_l = open(init_label_path,"a+",encoding="utf8")

    f_r_s = open(rest_sentence_path,"a+",encoding="utf8")
    f_r_l = open(rest_label_path,"a+",encoding="utf8")

    f_re_s = open(ready_for_select_train_sentence_path,"r",encoding="utf8")
    f_re_l = open(ready_for_select_train_label_path,"r",encoding="utf8")

    list_sentence = f_re_s.readlines()
    list_label = f_re_l.readlines()
    for i in range(len(list_sentence)):
        if i in list_k:
            print(list_sentence[i],file=f_i_s,end="")
            print(list_label[i],file=f_i_l,end="")
        else:
            print(list_sentence[i],file=f_r_s,end="")
            print(list_label[i],file=f_r_l,end="")

# select()