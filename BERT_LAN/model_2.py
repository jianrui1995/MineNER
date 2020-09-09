# @Time: 2020/8/6 14:37
# @Author: R.Jian
# @Note: BERT+BiLSTM+LAN

import tensorflow as tf
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import transformers
from BERT_LAN.layers import *
from BERT_LAN.sublayers import *
import BERT_LAN.HPsetting as HP

class Model(tf.keras.Model):
    def __init__(self,isNotInit):
        super(Model,self).__init__()
        conf = transformers.BertConfig.from_json_file("model/chinese_L-12_H-768_A-12/config.json")
        if not isNotInit:
            self.bertmodel = transformers.TFBertModel.from_pretrained("bert-base-chinese",config=conf,cache_dir="my_catch")
        else:
            self.bertmodel = transformers.TFBertModel(conf)
        self.bilstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768,return_sequences=True),merge_mode="concat")
        self.LN_1 = LayerNormalization()
        self.bilstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True),merge_mode="concat")
        self.LN_2 = LayerNormalization()
        self.bilstm_3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True),merge_mode="concat")
        self.LN_3 = LayerNormalization()
        self.bilstm_4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True),merge_mode="concat")
        self.LN_4 = LayerNormalization()
        self.bilstm_5 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(768, return_sequences=True),merge_mode="concat")
        self.LN_5 = LayerNormalization()
        self.lan = LabelAttention(units=HP.LABEL_NUM)
        self.dropout = tf.keras.layers.Dropout(HP.DROP_OUT_RATE)
        # 部分参数
        self.isTrainBert = tf.Variable(True,trainable=False)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        '''

        :param inputs: [x,mask] x: [timestep],tf.int32; mask:[timestep],tf.int32;
        :param training:表示是否训练模式
        :param mask:
        :return: [batch,timestep,labelnum],tf.floate32
        '''

        out_bert = self.bertmodel(inputs,training=False,output_hidden_states=False)
        # out = tf.add(out_bert[2][7],out_bert[2][7])

        out = out_bert[0]
        if not self.isTrainBert:
            out = tf.stop_gradient(out)

        # 第一层
        out_encoder = self.bilstm_1(out,mask=tf.cast(inputs[1],tf.bool))
        out_lan = self.lan(self.dropout(self.LN_1(out_encoder),training=training))
        out = tf.add(out_encoder,out_lan)
        # 第二层
        out_encoder = self.bilstm_2(out,mask=tf.cast(inputs[1],tf.bool))
        out_lan = self.lan(self.dropout(self.LN_2(out_encoder),training=training))
        out = tf.add(out_encoder,out_lan)
        # 第三层
        out_encoder = self.bilstm_3(out,mask=tf.cast(inputs[1],tf.bool))
        out_lan = self.lan(self.dropout(self.LN_3(out_encoder),training=training))
        out = tf.add(out_encoder, out_lan)
        # 第四层
        out_encoder = self.bilstm_4(out,mask=tf.cast(inputs[1],tf.bool))
        out_lan = self.lan(self.dropout(self.LN_4(out_encoder),training=training))
        out = tf.add(out_encoder, out_lan)
        # 第五层
        out_encoder = self.bilstm_5(out,mask=tf.cast(inputs[1],tf.bool))
        alpha = self.lan(self.dropout(self.LN_5(out_encoder),training=training),isfinal=True)
        # for metric
        predict = self.prepare_for_metric(alpha, inputs[1])

        return [alpha,predict]
        # return out_bert

    def prepare_for_metric(self,inputs,mask):
        '''
        这个是为方便metric的输入而准备的
        :param inputs: size:[batch,timestep,type_num],tf.float32
        :param mask: size:[batch,timeste],tf.int32
        :return: [batch,timestep],tf.int32
        '''
        out = tf.math.argmax(inputs,output_type=tf.int32,axis=-1)
        out = tf.math.multiply(out,mask)
        return out

def select(var):
    name = var.name
    dict_lr = {
        0.01 :["/bert/embeddings"]+["/bert/encoder/layer_._"+str(i) for i in range(0,6)],
        0.02 :["/bert/encoder/layer_._"+str(i) for i in range(6,9)],
        0.1 :["/bert/encoder/layer_._"+str(i) for i in range(9,12)] +["/bert/pooler"],
        1 : ["bidirectional","layer_normalization","label_attention"]
    }
    for k,v in dict_lr.items():
        for title in v:
            if name.find(title) != -1:
                return k
    return 1


if __name__ == "__main__":
    tokenizer = transformers.BertTokenizer("model/chinese_L-12_H-768_A-12/vocab.txt")
    text_2 = tokenizer.batch_encode_plus(["你买啊，买了就是成都人", "你来啊，来了就是深圳人"], max_length=20, pad_to_max_length=True)
    print(text_2)
    model = Model(True)
    out = model([tf.convert_to_tensor(text_2["input_ids"]),tf.convert_to_tensor(text_2['attention_mask'])])
    for data in model.trainable_variables:
        print(select(data))
        print(data.name)