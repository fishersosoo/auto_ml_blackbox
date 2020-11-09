import numpy as np


class PretrainTokenizer():
    """
    使用预训练的字向量将文本转化为向量
    """

    def __init__(self, path, topn=0):
        """

        Args:
            path: 字向量
            topn:
        """
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
        self.vectors = vectors
        self.iw = iw
        self.wi = wi
        self.dim = dim

    def word2vec(self, text, max_len, oov_vec=None, padding_vec=None):
        """
        将文本中的每个字或转化为向量
        Args:
            text (str):
            max_len (int):

        Returns:
            embedding: np array float [max_len, dim]
            mask: np array int [max_len]
        """
        if oov_vec is None:
            oov_vec = [0.] * self.dim
        if padding_vec is None:
            padding_vec = [0.] * self.dim
        embeddings = []
        if len(text) > max_len:
            max_index = max_len
        else:
            max_index = len(text)
        for char in text[:max_index]:
            embeddings.append(self.vectors.get(char, oov_vec))
        while len(embeddings) < max_len:
            embeddings.append(padding_vec)
        mask = [1] * max_index + [0] * (max_len - max_index)
        return np.array(embeddings), np.array(mask,dtype=float)


class FullTokenizer():
    def __init__(self):
        pass