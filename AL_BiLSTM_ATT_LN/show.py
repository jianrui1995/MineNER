# @Time: 2020/7/17 9:25
# @Author: R.Jian
# @Note: 

import visualdl

with open("data/loss.txt", "r", encoding="utf8") as fl, open(
        "data/score.txt", "r", encoding="utf8") as fs:
    out = [(float(l),float(s)) for l,s in zip(fl.readlines(),fs.readlines())]
    out = sorted(out,key=lambda x:x[0])
logwriter_l = visualdl.LogWriter("log/loss")
logwriter_s = visualdl.LogWriter("log/score")
for i in range(len(out)):
    logwriter_l.add_scalar("show/com",out[i][0],step=i)
    logwriter_s.add_scalar("show/com",out[i][1],step=i)