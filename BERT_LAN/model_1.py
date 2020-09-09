# @Time: 2020/7/21 16:45
# @Author: R.Jian
# @Note:

import tensorflow as tf
import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import transformers
from BERT_LAN.layers import *
import BERT_LAN.HPsetting as HP

class Model(tf.keras.Model):
    def __init__(self,isNotInit):
        super(Model,self).__init__()
        conf = transformers.BertConfig.from_json_file("model/chinese_L-12_H-768_A-12/config.json")
        if not isNotInit:
            self.bertmodel = transformers.TFBertModel.from_pretrained("bert-base-chinese",config=conf,cache_dir="my_catch")
        else:
            self.bertmodel = transformers.TFBertModel(conf)
        self.encoder1 = Encoder(HP.HEADS_NUM,HP.MASK_NUM)
        self.encoder2 = Encoder(HP.HEADS_NUM, HP.MASK_NUM)
        self.encoder3 = Encoder(HP.HEADS_NUM, HP.MASK_NUM)

        self.lan = LabelAttention(units=HP.LABEL_NUM)

        self.dropout = tf.keras.layers.Dropout(HP.DROP_OUT_RATE)
        # 部分参数
        self.isTrainBert = tf.Variable(False,trainable=False,name="isTrainBert")

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
        out_encoder = self.encoder1([out, inputs[1]], training=training)
        out_lan = self.lan(self.dropout(out_encoder,training=training))
        out = tf.add(out,out_lan)
        # 第二层
        out_encoder = self.encoder2([out,inputs[1]],training=training)
        out_lan = self.lan(self.dropout(out_encoder,training=training))
        out = tf.add(out,out_lan)
        # 第三层
        out_encoder = self.encoder3([out, inputs[1]], training=training)
        out_lan = self.lan(self.dropout(out_encoder,training=training))
        out = tf.add(out,out_lan)

        # for metric
        alpha = self.lan(self.dropout(out,training=training),isfinal=True)
        predict = self.prepare_for_metric(alpha,inputs[1])


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


if __name__ == "__main__":
    tokenizer = transformers.BertTokenizer("model/chinese_L-12_H-768_A-12/vocab.txt")
    text_2 = tokenizer.batch_encode_plus(["你买啊，买了就是成都人", "你来啊，来了就是深圳人"], max_length=20, pad_to_max_length=True)
    print(text_2)
    model = Model(True)
    out = model([tf.convert_to_tensor(text_2["input_ids"]),tf.convert_to_tensor(text_2['attention_mask'])])
    for data in model.variables:
        print(data.name)

