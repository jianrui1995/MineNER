import os
import BiLSTM_CRF
import dataset
import util
import setting
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''


u = util.util(setting.WORD2VEC_PATH)
d = dataset.Dateset()
model = BiLSTM_CRF.model()

restore_variable = model.ema.variables_to_restore()

saver = tf.train.Saver(restore_variable)

with tf.Session() as sess:
    x, y, l = u.word2vec(*d.getdata(setting.TRAIN_DATA, setting.BATCH_SIZE))
    ckpt = tf.train.get_checkpoint_state(setting.CKPT_PATH)
    saver.restore(sess,ckpt.model_checkpoint_path)
    decode_tag = sess.run(
        [model.acc],
        {
            model.x: x,
            model.y: y,
            model.seqence_length: l
        }
    )

    conut = 0
    for i in range(len(y)):
        for j in range(l[i]):
            if y[i][j] != 0:
                if decode_tag[0][i][j] == y[i][j]:
                    conut = conut + 1
    total = 0
    for i in l:
        total = total + 1
    print("正确率：==", conut / total)

