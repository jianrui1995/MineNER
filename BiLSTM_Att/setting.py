# @Time    : 2020/2/9 22:17
# @Author  : R.Jian
# @NOTE    : 模型训练的相关设置

# 模型存储的序号
STRAT_NUM = 1
# 训练的次数
EPOCH =10000
# 预测结果的存储路径：
TREAIN_SAVE_PATH = "result.txt"



# 模型的存储路径
MODEL_PATH_SAVE = "../model/bilstm-att_1/"
# 模型保存的名字
MODEL_NAME_SAVE = "bilstm_att"
# 模型的读取路径
MODEL_PATH_RESTORE = "../model/bilstm-att_1/"
# 模型读取的名字
MODEL_NAME_RESTORE = "bilstm_att"

# 训练句子的路径
PRO_TRAIN_SENTENCE_PATH = '../MNLP_Select/data/new_train_sentence.txt'
# 训练句子的标签路径
PRO_TRAIN_LABEL_PATH = '../MNLP_Select/data/new_train_label.txt'
# 训练句子的字数量路径
PRO_TRAIN_SUM_PATH = '../MNLP_Select/data/new_train_sum.txt'

# 测试句子的标签路径
PRO_TEST_LABEL_PATH = '../Data/CCKS2019_subtask1/programed/test_label.txt'
# 测试集句子的路径
PRO_TEST_SENTENCE_PATH = '../Data/CCKS2019_subtask1/programed/test_sentence.txt'
# 测试句子的字数量路径
PRO_TEST_SUM_PATH = '../Data/CCKS2019_subtask1/programed/test_sum.txt'
# 定义导入的数据类型
LOAD_PATH = [PRO_TRAIN_SENTENCE_PATH,PRO_TRAIN_SUM_PATH,PRO_TRAIN_LABEL_PATH]

# log的路径
LOG_PATH = "log.txt"
# 预定义的F1值
F1 = 0.8
# 训练多少次存储
SAVED_EVERY_TIMES = 25
# 批大小
BATCH_SIZE = 10
# LSTM_1 的神经元个数
LISM_1_UNTIS = 150
# LSTM_2 的神经元个数
LISM_2_UNTIS = 150
# LSTM_3 的神经元个数
LISM_3_UNTIS = 150

