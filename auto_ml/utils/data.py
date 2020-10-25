# coding=utf-8
import pandas as pd
import os
import json
import numpy as np

class TNewsData:
    def __init__(self):
        pass

    @staticmethod
    def _from_json(path):
        data = dict()
        with open(path, encoding='utf-8')as fd:
            text = fd.read()
            lines = text.split('\n')
            for line in lines:
                if len(line) < 2:
                    continue
                for k, v in json.loads(line).items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)
        df = pd.DataFrame(data)
        return df

    @staticmethod
    def read_source_dir(dir):
        dev_set = TNewsData._from_json(os.path.join(dir, "dev.json"))
        test_set = TNewsData._from_json(os.path.join(dir, "models.json"))
        train_set = TNewsData._from_json(os.path.join(dir, "train.json"))
        labels = TNewsData._from_json(os.path.join(dir, "labels.json"))
        return train_set, test_set, dev_set, labels


def read_vectors(path, topn=0):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim

def encode(vectors,text):
    embeddings=[]
    for char in text:
        embeddings.append(vectors[char])
    return embeddings

class encoder():
    def __init__(self,path,topn=0):
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        with open(path, encoding='utf-8', errors='ignore') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    dim = int(line.rstrip().split()[1])
                    continue
                lines_num += 1
                tokens = line.rstrip().split(' ')
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0])
                if topn != 0 and lines_num >= topn:
                    break
        for i, w in enumerate(iw):
            wi[w] = i
        self.vectors=vectors
        self.iw=iw
        self.wi=wi
        self.dim=dim

    def encode(self,text,max_len):
        embeddings = []
        for char in text[:max_len]:
            embeddings.append(self.vectors.get(char,default=0.))
        for len(embeddings)<max_len:
            embeddings.append([])



if __name__ == '__main__':
    # train_set, test_set, dev_set, labels = TNewsData.read_source_dir(
    #     "data/source/chineseGLUEdatasets.v0.0.1/tnews_public")

    vectors,iw, wi, dim=read_vectors(r"Z:\auto_ml\models\sgns.renmin.char")
    embeddings=encode(vectors,"史丹福")

