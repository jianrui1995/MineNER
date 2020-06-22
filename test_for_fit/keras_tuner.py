# @Time: 2020/6/17 19:21
# @Author: R.Jian
# @Note: keras-tunner的测试

import tensorflow as tf
from kerastuner.tuners import RandomSearch
from test_for_fit.model import model
def build_model():
    units = []