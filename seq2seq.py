from sklearn import preprocessing

from word2vector import word2vector


class seq2seq(object):

    def __init__(self, data=None):
        self.data = data
        self.get_data()

    def get_data(self):
        data = [
        ]

        f = open("./static/answer")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = f.readline()
            data.append(line)
        f.close()
        f = open("./static/question")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            line = f.readline()
            data.append(line)
        f.close()
        word2vec = word2vector(vector_length=100)
        word2vec.train(data)
        vocabulary = word2vec.vocabulary
        vocabulary_word = {}
        vocabulary_index = {}
        for item in vocabulary:
            current = vocabulary[item]
            huffman_vector = preprocessing.normalize(current['huffman_vector'])
            index = current['index']
            word_current = node(huffman_vector=huffman_vector, word=item, index=index)
            vocabulary_word[item] = word_current
            vocabulary_index[index] = word_current

        self.vocabulary_word = vocabulary_word
        self.vocabulary_index = vocabulary_index


class node(object):
    def __init__(self,
                 huffman_vector=None,
                 word=None,
                 index=None,
                 ):
        self.huffman_vector = huffman_vector
        self.word = word
        self.index = index


if __name__ == '__main__':
    seq2seq = seq2seq(None)
