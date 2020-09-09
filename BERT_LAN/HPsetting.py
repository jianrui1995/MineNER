# @Time: 2020/7/13 15:32
# @Author: R.Jian
# @Note: 

""
"模型参数相关"
# 头的数量
HEADS_NUM = 8
# attention中的mask的长度
MASK_NUM = 20
# encoder中全连接层中间层的维度
FULL_CONNETECTED_HIDDEN_DIM = 2048
# 词向量的维度
WORD_EMBEDDING_DIM = 768
# dropout比率
DROP_OUT_RATE = 0.1

"数据相关"
# label的数量
LABEL_NUM = 9
# sentence的路径
SENTENCE_PATH = "DataAnalysis-BIO-fix/two_train_sentence.txt"
# label的路径
LABEL_PATH = "DataAnalysis-BIO-fix/two_train_label.txt"
# validation的数据位置
VAL_SENTENCE_PATH = "DataAnalysis-BIO-fix/val_token.txt"
# vali结果文件
VAL_RESULT_PATH = "DataAnalysis-BIO-fix/val_result.txt"

"训练相关"
# 批大小
BATCH_SIZE = 20
# 学习率
LEARN_RATE = 0.0005
# 初始epoch
INITEPOCH = 0
# 结束epoch
EPOCH = 200
# 分层学习率 model_2
LR_WIED_LAYER = {
            0.01: ["/bert/embeddings"] + ["/bert/encoder/layer_._" + str(i) for i in range(0, 6)],
            0.02: ["/bert/encoder/layer_._" + str(i) for i in range(6, 9)],
            0.1: ["/bert/encoder/layer_._" + str(i) for i in range(9, 12)] + ["/bert/pooler"],
            1: ["bidirectional", "layer_normalization", "label_attention"]
        }
# model_1
# LR_WIED_LAYER = {
#     0.01: ["/bert/embeddings"] + ["/bert/encoder/layer_._" + str(i) for i in range(0, 6)],
#     0.02: ["/bert/encoder/layer_._" + str(i) for i in range(6, 9)],
#     0.1: ["/bert/encoder/layer_._" + str(i) for i in range(9, 12)] + ["/bert/pooler"],
#     1: ["encoder/create__heads", "encoder/merge__heads", "encoder_1","encoder_2","label_attention"]
# }

"存储&读取相关"
# 多少次训练一次存储
SAVED_EVERY_TIMES = 5
# 模型的存储路径
MODEL_PATH_SAVE = "model/Bert_Bi_BIO_fix_3/"
# 模型保存的名字
MODEL_NAME_SAVE = "model"
# 模型存储的序号
STRAT_NUM = 0
# 模型的读取路径
MODEL_PATH_RESTORE = "model/Bert_Bi_BIO_fix_3/"
# 模型读取的名字
MODEL_NAME_RESTORE = "model"
# 模型读取的序号
# MODEL_NUM_RESTORE = None
MODEL_NUM_RESTORE = MODEL_PATH_RESTORE+MODEL_NAME_RESTORE+"-165"

"可视化相关"
# visualdl的存储位置
LOG_DIR = "log-Bert_Bi_BIO_fix_3"
