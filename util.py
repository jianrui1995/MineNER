import gensim
import setting
import dataset
import numpy as np

class util():
    def __init__(self,word2vecpath):
        self.word2vecmodel = gensim.models.Word2Vec.load(word2vecpath)
        self.unknow = [0 for _ in range(300)]

    def word2vec(self,x,y):
        length = []
        maxlen = max(map(len,x))
        data = np.zeros([len(x),maxlen,setting.VEC_NUM],np.float32)
        for i in range(len(x)):
            length.append(len(x[i]))
            vec = [self.word2vecmodel[char] if char in self.word2vecmodel else self.unknow for char in x[i]]
            data[i,:len(x[i])] = vec
        return data,np.array(y),np.array(length)

    def test(self,word):
        if word not in self.word2vecmodel:
            print("AAA")
        else:
            print(self.word2vecmodel[word])


if __name__ == "__main__":
    u = util(setting.WORD2VEC_PATH)
    D = dataset.Dateset()
    a = D.getdata(setting.TEST_DATA)
    x,y,l = u.word2vec(*a)
    print(l)
