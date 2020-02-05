import tensorflow as tf
from BiLSTM_CRF import dataset, setting, util


class model():
    def __init__(self):
        # 导入占位符和变量的过程：
        self.creat_placeholders()
        self.creat_variable()
        # 创建指数平滑
        self.ema = self.ema()
        self.loss = self.predict()
        self.train_op = self.optimize()
        self.acc  = self.acc()

    def creat_placeholders(self):
        self.x = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None, setting.VEC_NUM],
            name="x"
        )
        self.y = tf.placeholder(
            dtype=tf.int32,
            shape=[None,None],
            name="y"
        )
        self.seqence_length = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name = "seqence_length"
        )

    def creat_variable(self):
        self.w = tf.get_variable(
            name="w",
            shape=[2 * setting.HIDDEN_SIZE, setting.TAG_NUM],  # 标签数目还需要改，只是随便写了一个值。
            dtype=tf.float32
        )
        self.b = tf.get_variable(
            name="b",
            shape=[1, setting.TAG_NUM]
        )
        self.global_step = tf.Variable(0,trainable=False)


    def predict(self):
        '''
        向前传播过程
        :return:
        '''
        # Bi_LSTM过程：
        x = tf.nn.dropout(self.x, setting.DROP_INPUT_KEEP)
        lstm_b = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
            setting.HIDDEN_SIZE), setting.DROP_INPUT_KEEP) for _ in range(setting.NUM_LAYER)])
        lstm_f = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
            setting.HIDDEN_SIZE), setting.DROP_INPUT_KEEP) for _ in range(setting.NUM_LAYER)])
        (output_f,output_b) ,_= tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_f,
            cell_bw=lstm_b,
            inputs=self.x,
            sequence_length=self.seqence_length,
            dtype=tf.float32
        )
        output = tf.concat([output_f,output_b],axis=-1)
        output = tf.nn.dropout(output, setting.DROP_INPUT_KEEP) # 3维的张量。
        s = tf.shape(output)
        output = tf.reshape(output, shape=[-1, 2 * setting.HIDDEN_SIZE])

        # 全连接层
        output = tf.matmul(output,self.w)+self.b
        self.output = tf.reshape(output, [-1, s[1], setting.TAG_NUM])

        # CRF过程：
        log_likelihood,self.trans_params = tf.contrib.crf.crf_log_likelihood(
            inputs=self.output,
            tag_indices = self.y,
            sequence_lengths = self.seqence_length
        )

        # 损失
        loss = -tf.reduce_mean(log_likelihood)

        return loss

    def ema(self):
        '''
        指数平滑
        :return:
        '''
        ema = tf.train.ExponentialMovingAverage(setting.EMA_RATE, self.global_step)
        return ema

    def optimize(self):
        '''
        定义反向传播
        :return:
        '''
        learn_rate = tf.train.exponential_decay(setting.LEARN_RATE, self.global_step, setting.LR_DECAY_STEP,
                                                setting.LR_DECAY)

        optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss,self.global_step)
        ave_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([optimizer, ave_op]):
            train_op = tf.no_op('train')
            return train_op

    def acc(self):
        '''
        验证测试集
        :return:
        '''
        decode_tag,_= tf.contrib.crf.crf_decode(
            self.output,
            self.trans_params,
            self.seqence_length
        )

        return decode_tag


class train():
    def __init__(self):
        self.model = model()
        self.u = util.util(setting.WORD2VEC_PATH)
        self.d = dataset.Dateset()


    def start_train(self):
        current_loss = 1000
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(setting.TRAIN_TIMES):

                x,y,l = self.u.word2vec(*self.d.getdata(setting.TEST_DATA, setting.BATCH_SIZE))
                loss ,_ = sess.run([self.model.loss,self.model.train_op],
                                   {
                                       self.model.x:x,
                                       self.model.y:y,
                                       self.model.seqence_length:l
                                   })
                # 输出loss
                if step % setting.SHOW_STEP == 0:
                    print("step = {}, loss = {}".format(step,loss))
                    # 如果损失值小于最小值那么模型保存
                    if loss < current_loss:
                        loss = current_loss
                        saver.save(sess, setting.CKPT_PATH + setting.MODEL_NAME, self.model.global_step)
