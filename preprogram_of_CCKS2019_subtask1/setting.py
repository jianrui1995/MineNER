# 原始训练集的路径
ORI_TRATIN_PATH = '../Data/CCKS2019_subtask1/ori/subtask1_training_part2.txt'
# 原始测试集的路径
ORI_TEST_PATH = '../Data/CCKS2019_subtask1/ori/subtask1_training_part1.txt'

# 训练句子的路径
PRO_TRAIN_SENTENCE_PATH = '../Data/CCKS2019_subtask1/programed/train_sentence.txt'
# 测试集句子的路径
PRO_TEST_SENTENCE_PATH = '../Data/CCKS2019_subtask1/programed/test_sentence.txt'
# 训练句子的标签路径
PRO_TRAIN_LABEL_PATH = '../Data/CCKS2019_subtask1/programed/train_label.txt'
# 测试句子的标签路径
PRO_TEST_LABEL_PATH = '../Data/CCKS2019_subtask1/programed/test_label.txt'
# 训练句子的字数量路径
PRO_TRAIN_SUM_PATH = '../Data/CCKS2019_subtask1/programed/train_sum.txt'
# 测试句子的字数量路径
PRO_TEST_SUM_PATH = '../Data/CCKS2019_subtask1/programed/test_sum.txt'




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