# @Time    : 2020/2/11 12:39
# @Author  : R.Jian
# @NOTE    : 测试tensorshape


import numpy as np

list_a = np.array([1,1,0])
for i in range(len(list_a)):
    if list_a[i] != 0:
        print(list_a[i])

