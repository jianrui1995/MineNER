# @Time: 2020/7/9 16:30
# @Author: R.Jian
# @Note: 

import sys
sys.path.append(r"/home/tech/myPthonProject/MineNER/")
import tensorflow as tf

from AL_BiLSTM_ATT_LN.model import Model
from AL_BiLSTM_ATT_LN.datasets import Out,Out_For_Predict
from AL_BiLSTM_ATT_LN.metrics import F1
from AL_BiLSTM_ATT_LN.losses import LossForLMP,LossForTask,Loss_Return_0
import AL_BiLSTM_ATT_LN.HPsetting as setting
from AL_BiLSTM_ATT_LN.callbacks import Save,VisualDL,isUseLL,ForPredict

def train():
    train_dataset = Out(*setting.LOAD_NEW_PATH)().shuffle(200,seed=setting.SEED_NUM).padded_batch(
                                                            batch_size=setting.BATCH_SIZE,
                                                            padded_shapes=(([None,400],[None],[None,13]),([None],[None],[None],[None,None])),
                                                            padding_values=((0.0,False,0),(0,0,0,0))
                                                               ).prefetch(3)
    test_dataset = Out(*setting.LOAD_TEST_PATH)().padded_batch(
                                                            batch_size=setting.BATCH_SIZE,
                                                            padded_shapes=(([None,400],[None],[None,13]),([None],[None],[None],[None,None])),
                                                            padding_values=((0.0,False,0),(0,0,0,0))
                                                               ).prefetch(3)
    model = Model()
    # for data in train_dataset.take(1):
    #     # print(data)
    #     print(model(data[0],training=True))


    def LossRaterScheduler(epoch):
        if epoch+1> setting.STOP_NUM:
            return setting.LEARN_RATE
        else:
            return setting.LEARN_RATE

    # callbacks的创建
    learnrateschedulder = tf.keras.callbacks.LearningRateScheduler(LossRaterScheduler)
    save = Save(setting.SAVED_EVERY_TIMES,setting.MODEL_PATH_SAVE,setting.MODEL_NAME_SAVE,setting.MODEL_NUM_RESTORE)
    visual = VisualDL("log/",setting.SAVED_EVERY_TIMES)
    isusell = isUseLL(setting.STOP_NUM)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(setting.LEARN_RATE),
        loss=[LossForTask(),LossForLMP(),Loss_Return_0(),Loss_Return_0()],
        metrics=[[],[],[],[F1()]],
        loss_weights=[1,1,0,0]
    )
    model.fit(
        x=train_dataset,
        initial_epoch=setting.INIT_EPOCH,
        epochs=setting.EPOCH,
        validation_freq=setting.SAVED_EVERY_TIMES,
        validation_data=test_dataset,
        callbacks=[save,visual,learnrateschedulder,isusell]
    )

def predict():
    test_dataset = Out_For_Predict(*setting.LOAD_OLD_PATH)().padded_batch(
        batch_size=30,
        padded_shapes=(([None,400],[None],[None,None]),([None,None])),
        padding_values=((0.0,False,0),(0))
    ).prefetch(3)

    f = open("data/score.txt", "w", encoding="utf8")
    f2 = open("data/loss.txt", "w", encoding="utf8")
    model = Model()
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(setting.MODEL_NUM_RESTORE)
    model.ispredict.assign(True)
    model.isUseLL.assign(True)
    for data in test_dataset:
        out = model(data[0],training=False)
        loss=out[0].numpy()
        score = out[2].numpy()
        for i in range(len(loss)):
            print(str(loss[i]),file=f2)
            print(str(score[i][0]),file=f)
    f.close()
    f2.close()


if setting.M == "train":
    train()
elif setting.M == "predict":
    predict()


