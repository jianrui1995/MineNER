# @Time: 2020/7/13 15:01
# @Author: R.Jian
# @Note: 存在几个问题需要解决，1）多线程，一旦输入之后，attention_mask无法在后面的任务进行使用。只找到一种方法，在修改bert层输出后，解决、
# 2）还是因为多线程的原因，还有一个错误，不知道怎么描述，就这样吧，不然代码写到啥时候。去啊

import tensorflow as tf
import transformers
from BERT_LAN.layers import *
import BERT_LAN.HPsetting as HP

# transformers.TFBertModel
# transformers.TFBertForSequenceClassification


class Model(transformers.TFBertPreTrainedModel):
    def __init__(self,config, *inputs, **kwargs):
        super(Model,self).__init__(config, *inputs, **kwargs)
        self.bert = transformers.TFBertMainLayer(config,name="bert")
        self.encoder = Encoder(HP.HEADS_NUM,HP.MASK_NUM)


    @tf.function
    def call(self, inputs, training=None, mask=None,**kwargs):

        out = self.bert(inputs)
        out = self.encoder([out[0],inputs[1]])
        return out


if __name__ == "__main__":

    tokenizer = transformers.BertTokenizer("model/chinese_L-12_H-768_A-12/vocab.txt")
    text_2 = tokenizer.batch_encode_plus(["你买啊，买了就是成都人", "你来啊，来了就是深圳人"], max_length=20, pad_to_max_length=True)
    print(text_2)
    conf = transformers.BertConfig().from_json_file("model/chinese_L-12_H-768_A-12/config.json")
    model = Model.from_pretrained("bert-base-chinese")

    # 测试的模型
    # model = transformers.TFBertModel.from_pretrained("bert-base-chinese")
    # model = transformers.TFBertForSequenceClassification.from_pretrained("bert-base-chinese")

    out = model([tf.convert_to_tensor(text_2["input_ids"]),tf.convert_to_tensor(text_2['attention_mask'])])
    print("out",out)

    # input = tf.ones([2,50,768],dtype=tf.float32)
    # mask = tf.ones([2,50],dtype=tf.int32)
    # model = Model(conf)
    # output = model([input,mask])
    # print(output)