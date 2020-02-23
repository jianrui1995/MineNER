# 原始训练集的路径
ORI_TRATIN_PATH = '../Data/CCKS2019_subtask1/ori/subtask1_training_part1.txt'
# 原始测试集的路径
ORI_TEST_PATH = '../Data/CCKS2019_subtask1/ori/subtask1_testing_part2.txt'
# 训练句子的路径
PRO_TRAIN_SENTENCE_PATH = '../Data/CCKS2019_subtask1/programed/train_sentence.txt'
# 训练句子的标签路径
PRO_TRAIN_LABEL_PATH = '../Data/CCKS2019_subtask1/programed/train_label.txt'
# 训练句子的字数量路径
PRO_TRAIN_SUM_PATH = '../Data/CCKS2019_subtask1/programed/train_sum.txt'
# 测试句子的标签路径
PRO_TEST_LABEL_PATH = '../Data/CCKS2019_subtask1/programed/test_label.txt'
# 测试集句子的路径
PRO_TEST_SENTENCE_PATH = '../Data/CCKS2019_subtask1/programed/test_sentence.txt'
# 测试句子的字数量路径
PRO_TEST_SUM_PATH = '../Data/CCKS2019_subtask1/programed/test_sum.txt'
# WORD2VEC模型的路径
WORD2VEC_PATH = '../model/word2vec/word2voc_char_size300_iter140.model'
# GLOVE模型的路径
GLOVE_PATH = "model/glove/vectors.txt"
# 设置句子的最大长度，又程序自己计算
MAX_SUM = 0
# 设置字向量的向量大小
WORDVEC_NUM = 300
# 设置位置向量的大小
LOCVEC_NUM = 100
# 向量的总长度
TOTALVEC_NUM = WORDVEC_NUM+LOCVEC_NUM
# 位置向量的持久化路径
LOC_VEC_PATH = "../Data/CCKS2019_subtask1/programed/loc_vec.json"


# 定义导入的数据类型
LOAD_PATH = [PRO_TRAIN_SENTENCE_PATH,PRO_TRAIN_SUM_PATH,PRO_TRAIN_LABEL_PATH]


# 标签和实体名称的对应关系
NAME2TAG={
            "疾病和诊断":"dis",
            "手术":"ope",
            "影像检查":"YXche",
            "解剖部位":"bod",
            "药物":"mad",
            "实验室检验":"SYche"
        }
# 标签和代表数字的对应关系
TAG2LABEL = {
            "o":0,
            "B_dis":1,
            "I_dis":2,
            "B_ope":3,
            "I_ope":4,
            "B_mad":5,
            "I_mad":6,
            "B_bod":7,
            "I_bod":8,
            "B_YXche":9,
            "I_YXche":10,
            "B_SYche":11,
            "I_SYche":12
        }
# 设置label的总量
LABEL_SUM = len([i for i in TAG2LABEL.items()])