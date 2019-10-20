import gensim
import setting
import dataset

class util():
    def __init__(self,word2vecpath):
        self.word2vecmodel = gensim.models.Word2Vec.load(word2vecpath)
        self.unknow = [0 for _ in range(300)]

    def word2vec(self,x,y):
        vec_x = []
        for chars in x:
            vec = [self.word2vecmodel[char] if char in self.word2vecmodel else self.unknow for char in chars]
            vec_x.append(vec)
        return vec_x,y

    def test(self,word):
        if word not in self.word2vecmodel:
            print("AAA")
        else:
            print(self.word2vecmodel[word])


if __name__ == "__main__":
    u = util(setting.WORD2VEC_PATH)
    D = dataset.Dateset()
    a = D.getdata(setting.TEST_DATA)
    x,y = u.word2vec(*a)
    print(len(x))
