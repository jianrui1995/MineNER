# @Time: 2020/2/21 14:55
# @Author: R.Jian
# @Note: MNLP的样本选择算法的参数设置

# 训练句子的路径
PRO_TRAIN_SENTENCE_PATH = '../Data/CCKS2019_subtask1/programed/train_sentence.txt'
# 训练句子的标签路径
PRO_TRAIN_LABEL_PATH = '../Data/CCKS2019_subtask1/programed/train_label.txt'
# 训练句子的字数量路径
PRO_TRAIN_SUM_PATH = '../Data/CCKS2019_subtask1/programed/train_sum.txt'

# 摘取的训练句子的路径
PRO_TRAIN_SENTENCE_NEW_PATH = 'data/new_train_sentence.txt'
# 摘取训练句子的标签路径
PRO_TRAIN_LABEL_NEW_PATH = 'data/new_train_label.txt'
# 摘取训练句子的字数量路径
PRO_TRAIN_SUM_NEW_PATH = 'data/new_train_sum.txt'

# 摘取的训练句子的路径
PRO_TRAIN_SENTENCE_OLD_PATH = 'data/old_train_sentence.txt'
# 摘取训练句子的标签路径
PRO_TRAIN_LABEL_OLD_PATH = 'data/old_train_label.txt'
# 摘取训练句子的字数量路径
PRO_TRAIN_SUM_OLD_PATH = 'data/old_train_sum.txt'

# dataset载入的数据
LOAD_PATH = [PRO_TRAIN_SENTENCE_OLD_PATH,PRO_TRAIN_SUM_OLD_PATH,PRO_TRAIN_LABEL_OLD_PATH]
# 载入的模型路径
MODEL_RESTORE_PATH = "../model/bilstm-att_1/"
# 模型的名字
MODEL_NAME = "bilstm_att-6"
# 模型的载入完整路径
MODEL_PATH = MODEL_RESTORE_PATH+MODEL_NAME

# 概率文件存放路径
PRO_SELETC_FILE_PATH = "data/old_select.txt"
# 选择前n小的概率
TOP_N = 100
