import json
import setting
import random

class Dateset():
    def __init__(self):
        self.tag2label = {
            "o":0,
            "B_dis":1,
            "I_dis":2,
            "B_ope":3,
            "I_ope":4,
            "B_mad":5,
            "I_mad":6,
            "B_bod":7,
            "I_bod":8,
            "B_YXche":9,
            "I_YXche":10,
            "B_SYche":11,
            "I_SYche":12
        }
        self.name2tag={
            "疾病和诊断":"dis",
            "手术":"ope",
            "影像检查":"YXche",
            "解剖部位":"bod",
            "药物":"mad",
            "实验室检验":"SYche"
        }
        f_test = open(setting.POS_TEST,"r",encoding="utf8")
        self.pos_test = json.load(f_test)
        f_train = open(setting.POS_TRAIN,"r",encoding="utf8")
        self.pos_train = json.load(f_train)

    def readorifile(self,path,type):
        '''
        对原始数据的载入函数，生成已经带标签的原始数据集。
        :param path:
        :param type:
        :return:
        '''
        with open(path,"r",encoding="utf8") as f1:
            list_lines = f1.readlines()
            list_total_x = []
            list_total_y = []
            for str_line in list_lines:
                if str_line.startswith(u"\ufeff"):
                    str_line = str_line.encode("utf8")[3:].decode("utf8")
                dict_ori = json.loads(str_line)
                list_x = list(dict_ori["originalText"])
                list_y = [0 for _ in range(len(list_x))]
                for entity in dict_ori["entities"]:
                    left = list_y[:entity["start_pos"]]
                    right = list_y[entity["end_pos"]:]
                    label = []
                    int_I = entity["end_pos"] - entity["start_pos"] -1
                    label.append(self.tag2label["B_"+self.name2tag[entity["label_type"]]])
                    label = label + [self.tag2label["I_"+self.name2tag[entity["label_type"]]] for _ in range(int_I)]
                    list_y = left + label + right
                list_total_x.append(list_x)
                list_total_y.append(list_y)
            all = {
                "x":list_total_x,
                "y":list_total_y
            }
        f1=open("Data/pos/"+type,"w",encoding="utf8")
        json.dump(all,f1,ensure_ascii=False)

    def getdata(self,type=1,size=100):
        '''
        用于抽取size大小的数据，作为模型输入。
        :param size: 抽取的数据大小。
        :param type: 抽取的数据集类型： 1：测试集   2：测试集
        :return: 返回数据集和结果元祖。
        '''
        if type == 1 :
            dict_data = self.pos_test
        if type == 2 :
            dict_data = self.pos_train

        list_sequence = random.sample(range(0,len(dict_data["x"])),size)
        list_x = [dict_data["x"][i] for i in list_sequence]
        list_y = [dict_data["y"][i] for i in list_sequence]
        return list_x,list_y


if __name__=="__main__":
    D = Dateset()
    # D.readorifile("Data/ori/subtask1_training_part1.txt","train.json")
    # D.readorifile("Data/ori/subtask1_training_part2.txt","test.json")
    data = D.getdata(setting.TEST_DATA)
    print(data)