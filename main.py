import BiLSTM_CRF
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
t = BiLSTM_CRF.train()
t.start_train()