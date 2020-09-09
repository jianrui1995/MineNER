# @Time: 2020/5/11 22:31
# @Author: R.Jian
# @Note: model_LN的超参数设置

# epoch的开始索引
INIT_EPOCH = 0
# epoch的结束索引
EPOCH =50
# 预测结果的存储路径：
TREAIN_SAVE_PATH = "result.txt"
# 学习率
LEARN_RATE = 1e-3

# 模式
M = "train"
# M = "predict"
# 模型的存储路径
MODEL_PATH_SAVE = "model/RANDOM-3000/"
# 模型保存的名字
MODEL_NAME_SAVE = "model"
# 模型存储的序号
STRAT_NUM = 0

# 模型的读取路径
MODEL_PATH_RESTORE = "model/RANDOM-2500/"
# 模型读取的名字
MODEL_NAME_RESTORE = "model"
# 模型读取的序号
# MODEL_NUM_RESTORE = None
MODEL_NUM_RESTORE = MODEL_PATH_RESTORE+MODEL_NAME_RESTORE+"-20"

# glove模型的位置
GLOVE_VEC_PATH = "model/glove/vectors.txt"
# 位置向量的位置
LOC_VEC_PATH = "data/ori/loc_vec.json"

# 训练句子的路径  选中
PRO_TRAIN_NEW_SENTENCE_PATH = 'data/init_train_sentence.txt'
# 训练句子的标签路径  选中
PRO_TRAIN_NEW_LABEL_PATH = 'data/init_train_label.txt'
# 定义导入的数据类型  选中
LOAD_NEW_PATH = [PRO_TRAIN_NEW_SENTENCE_PATH,GLOVE_VEC_PATH,LOC_VEC_PATH,PRO_TRAIN_NEW_LABEL_PATH]

# 训练句子的路径  准备用来选择的路径
PRO_TRAIN_OLD_SENTENCE_PATH = 'data/ready_for_select_train_sentence_path.txt'
# 训练句子的标签路径  准备用来选择的路径
PRO_TRAIN_OLD_LABEL_PATH = 'data/ready_for_select_train_label_path.txt'
# 定义导入的数据类型  准备用来选择的路径
LOAD_OLD_PATH = [PRO_TRAIN_OLD_SENTENCE_PATH,GLOVE_VEC_PATH,LOC_VEC_PATH,PRO_TRAIN_OLD_LABEL_PATH]

# 测试句子的标签路径
PRO_TEST_LABEL_PATH = 'data/ori/test_label.txt'
# 测试集句子的路径
PRO_TEST_SENTENCE_PATH = 'data/ori/test_sentence.txt'
# 定义测试集的路径
LOAD_TEST_PATH = [PRO_TEST_SENTENCE_PATH,GLOVE_VEC_PATH,LOC_VEC_PATH,PRO_TEST_LABEL_PATH]


# log的路径，存模型的文件夹有对应关系,用于给tensorBoard传入参数
LOG_DIR = "log/"
# 预定义的F1值
F1 = 0.8
# 训练多少次存储 和 训练多少次验证
SAVED_EVERY_TIMES = 10
# 批大小
BATCH_SIZE = 30
# 随机量
SEED_NUM = 6
# label的数量
LABEL_NUM =13
# LSTM_1 的神经元个数
LISM_1_UNTIS = 256
# LSTM_2 的神经元个数
LISM_2_UNTIS = 256
# LSTM_3 的神经元个数
LISM_3_UNTIS = 256

# 主动学习，dense_1的神经元个数
LL_DENSE_1 = 512
# 计算损失函数的边际量
THETA = 20
# 停止训练LPM模块,也在该EPOCH之后，学习率减少为十分之二。
STOP_NUM = 30
# 输出通道
FILTERS = 3
# 卷积核大小
KERNEL_SIZE = [5, 8]
# 步长
STRIDES = [1, 3]
