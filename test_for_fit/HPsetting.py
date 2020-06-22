# @Time: 2020/6/17 18:25
# @Author: R.Jian
# @Note: 超参数设置

from tensorboard.plugins.hparams import api as hp

hp_1=hp.HParam("dense_1",hp.Discrete([16,32]))
hp_2=hp.HParam("dense_2",hp.RealInterval(16,32))
hp_3=hp.HParam("dense_3",hp.RealInterval([16,32]))

