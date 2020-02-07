# @Time    : 2020/2/8 0:02
# @Author  : R.Jian
# @NOTE    : 
import preprogram_of_CCKS2019_subtask1.setting as setting
import numpy as np
import json
"随机生成距离向量"
loc_dic = {i: np.random.random(setting.LOCVEC_NUM).tolist() for i in range(0, 1000)}
f = open(setting.LOC_VEC_PATH, "w", encoding="utf8")
json.dump(loc_dic, f, ensure_ascii=False)
f.close()