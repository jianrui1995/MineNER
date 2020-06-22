# @Time: 2020/6/19 16:31
# @Author: R.Jian
# @Note: 

from visualdl import LogWriter

if __name__ == '__main__':
    value = [i/1000.0 for i in range(1000,2000)]
    # 步骤一：创建父文件夹：log与子文件夹：scalar_test

    for step in range(23,100):
        with LogWriter(logdir="./visualDL/scalar_test3") as writer:

        # 步骤二：向记录器添加一个tag为`train/acc`的数据
            writer.add_scalar(tag="train/acc", step=step, value=4)
        # 步骤二：向记录器添加一个tag为`train/loss`的数据
            writer.add_scalar(tag="train/loss", step=step, value=3)



