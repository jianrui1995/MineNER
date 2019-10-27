
# 源数据路径
POS_TEST = "Data/pos/test.json"
POS_TRAIN = "Data/pos/train.json"

# 提取一次训练集的大小。
BATCH_SIZE = 100

# 获取的数据来源于训练集
TEST_DATA = 1
# 获取的数据来源于测试集
TRAIN_DATA = 2

# word2vec词向量模型路径
WORD2VEC_PATH = "model/word2vec/word2voc_char_size300_iter140.model"

# 隐藏层神经元个数
HIDDEN_SIZE = 300

# drop 的 input保留率：
DROP_INPUT_KEEP = 0.5

# 隐藏层层数
NUM_LAYER = 2

#标签数目
TAG_NUM = 11

#词向量空间大小
VEC_NUM = 300

#指数平滑衰减率：
EMA_RATE = 0.99

# 学习率
LEARN_RATE = 0.0001

# 学习率衰减
LR_DECAY = 0.99

# 衰减频率
LR_DECAY_STEP = 1000