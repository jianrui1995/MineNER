# @Time: 2020/7/27 15:25
# @Author: R.Jian
# @Note: 
import sys
sys.path.append(r"/home/ruijian/myproject/MineNER/")
from BERT_LAN.datasets import *
import BERT_LAN.HPsetting as HP
from BERT_LAN.model_2 import Model
from BERT_LAN.losses import En_Cross
from BERT_LAN.metrics import F1,Precision,Recall
from BERT_LAN.optimazers import Adam_with_layer
from BERT_LAN.callbacks import Save,VisualDL,Load

def train():
    train_dataset = Out(HP.SENTENCE_PATH,HP.LABEL_PATH)().padded_batch(
        batch_size=HP.BATCH_SIZE,
        padded_shapes=(([None],[None]),[None,HP.LABEL_NUM]),
        padding_values=((0,0),0)
    )
    model = Model(isNotInit=HP.MODEL_NUM_RESTORE)
    # callback
    save = Save(HP.SAVED_EVERY_TIMES,HP.MODEL_PATH_SAVE,HP.MODEL_NAME_SAVE,HP.MODEL_NUM_RESTORE)
    visual = VisualDL(HP.LOG_DIR,HP.SAVED_EVERY_TIMES)
    model.compile(
        optimizer=Adam_with_layer(HP.LEARN_RATE,dict_lr_schedul=HP.LR_WIED_LAYER),
        loss=[En_Cross(),None],
        metrics=[[],[Precision(),Recall(),F1()]]
    )
    i= 1

    model.fit(
        x=train_dataset,
        initial_epoch=HP.INITEPOCH,
        epochs=HP.EPOCH,
        callbacks=[save,visual]
    )

def predict():
    val_dataset = Sentence(HP.VAL_SENTENCE_PATH)().padded_batch(
        batch_size=HP.BATCH_SIZE,
        padded_shapes=([None],[None]),
        padding_values=(0,0)
    )
    f = open(HP.VAL_RESULT_PATH, "w", encoding="utf8")
    model = Model(isNotInit=HP.MODEL_NUM_RESTORE)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(HP.MODEL_NUM_RESTORE)
    for data in val_dataset:
        out = model(data,training=False)
        pre = out[1].numpy()
        for da in pre:
            da = [str(i) for i in da]
            print(" ".join(da),file=f)
    f.close()


if __name__ == "__main__":
    # train()
    predict()