# @Time: 2020/2/27 14:16
# @Author: R.Jian
# @Note: 

import tensorflow as tf
import numpy as np

class ra():
    def __init__(self):
        self.x = np.random.random(size=(100,10))
        self.z = np.random.randint(0,3,size=(100))
        self.y = [[0,0,0] for _ in self.z]
        self.m = [[0, 0, 0] for _ in self.z]
        for i in range(len(self.z)):
            self.y[i][self.z[i]] = 1

    def __call__(self, *args, **kwargs):
        return tf.data.Dataset.from_tensor_slices((self.x,(self.y,self.y,self.y)))

tf.keras.callbacks.LearningRateScheduler
if __name__ == "__main__":
    r = ra()
    for data in r():
        print(data)
