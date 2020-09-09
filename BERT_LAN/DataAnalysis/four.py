# @Time: 2020/7/10 17:42
# @Author: R.Jian
# @Note:

import json

# HPsetting
three_train_info ="three_train_info_3.json"

def four(info):
    f = open(info,"r",encoding="utf8")
    result =json.load(f)
    for k,v in result.items():
        alone = []
        double = []
        for k_1,v_1 in v.items():
            v_2 = []
            for data in v_1:
                if data not in v_2:
                    v_2.append(data)
            for data in v_2:
                if data not in double:
                    if data not in alone:
                        alone.append(data)
                    else:
                        alone.remove(data)
                        double.append(data)
        if len(double)>0:
            print(len(alone))
            print(len(double))
            print()


four(three_train_info)