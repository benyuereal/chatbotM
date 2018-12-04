'''
:keyword word2vector
这是一个单词去向量的类
输入的是学习率、词典（有可能没有）、隐藏层大小、输入的数据
'''
from collections import Counter
import numpy as np
import jieba
from sklearn import preprocessing


class word2vector(object):
    def __init__(self,
                 learning_rate=0.025,
                 vocabulary_size=15000,
                 hidden_size=100,
                 input=None,
                 vector_length=100,
                 vocabulary=None,
                 window_length=5
                 ):
        self.learning_rate = learning_rate
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.input = input
        self.vector_length = vector_length
        self.vocabulary = vocabulary
        self.window_length = window_length
        self.huffman_tree = None

    # 将单词转化为词频字典
    def word_dictionary(self, input):
        word = []
        counter = {}
        for line in input:
            line = line.strip()
            if len(line) == 0:
                continue
            line = jieba.cut(line, cut_all=True)
            for w in line:
                if not w in word:
                    word.append(w)
                if not w in counter:
                    counter[w] = 1
                else:
                    counter[w] += 1
        data2 = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        vocabulary = dict(data2)
        return vocabulary

    '''
    训练的时候 按照行就行一组投送 这样的话就避免了不同对话产生的差距
    '''

    def train(self, input):
        # 产生词频统计
        self.vocabulary = self.word_dictionary(input)
        if self.huffman_tree == None:
            # 初始化哈夫曼编码等信息
            self.vocabulary = self.generate_vocabulary(self.vocabulary)
        self.huffman_tree = huffman(self.vocabulary, vector_length=self.vector_length)
        print('word_dict and huffman tree already generated, ready to train vector')
        # 设置待训练的单词在窗口中左右两边的单词长度,一般情况下窗口是奇数，比如 11 左边是5 右边也是5
        left = (self.window_length - 1) >> 1
        right = self.window_length - 1 - left
        for line in input:
            line = list(line.strip())
            line_len = line.__len__()
            for i in range(line_len):
                # 相当于拿到目标单词左右两边 窗口大小的单词量，有时候刚开始的时候目标单词左边是只有一个或者没有单词的
                self.deal_gram_CBOW(line[i], line[max(0, i - left):i] + line[i + 1:min(line_len, i + right + 1)])
        print('word vector has been generated')

    def deal_gram_CBOW(self, word, gram_word_list):
        # 如果单词不在字典表里面 那么肯定直接结束掉了
        if not self.vocabulary.__contains__(word):
            return
        # 当前这个单词的哈夫曼编码
        huffman_code = self.vocabulary[word]['huffman_code']
        # 暂时不理解这个
        gram_vector_sum = np.zeros([1, self.vector_length])
        #  这个是python的slice notation的特殊用法。
        #
        # a = [0,1,2,3,4,5,6,7,8,9]
        # b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
        # b = a[1:3] 那么，b的内容是 [1,2]
        # 当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
        # 当j缺省时，默认为len(alist), 即a[1:]相当于a[1:10]
        # 当i,j都缺省时，a[:]就相当于完整复制一份a了
        #
        # b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.
        # 所以a[i:j:1]相当于a[i:j]
        # 当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
        # fixme 所以a[::-1]相当于 a[-1:-len(a)-1:-1]，
        # fixme 也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。

        # 这一个步骤 相当于累加目标词两边的词向量
        for i in range(gram_word_list.__len__())[::-1]:
            item = gram_word_list[i]
            if self.vocabulary.__contains__(item):
                gram_vector_sum += self.vocabulary[item]['huffman_vector']
            else:
                gram_word_list.pop(i)

        if gram_word_list.__len__() == 0:
            return
        e = self.__GoAlong_Huffman(huffman_code, gram_vector_sum, self.huffman_tree.root)
        # 词向量更新
        for item in gram_word_list:
            # 词向量更新
            self.vocabulary[item]['huffman_vector'] += e
            self.vocabulary[item]['huffman_vector'] = preprocessing.normalize(
                self.vocabulary[item]['huffman_vector'])

    # 进行沿着哈夫曼树
    def __GoAlong_Huffman(self, huffman_code, input_vector, root):

        node = root
        e = np.zeros([1, self.vector_length])
        # 假如哈夫曼编码是 1001 也就是 '左右右左'。
        # 以下是每个节点
        for level in range(huffman_code.__len__()):
            huffman_charat = huffman_code[level]
            # 判别正类和负类的方法是使用sigmoid函数
            q = self.__Sigmoid(input_vector.dot(node.huffman_vector.T))
            # 梯度公式 ∂L∂xw=∑j=2lw(1−dwj−σ(xTwθwj−1))θwj−1
            grad = self.learning_rate * (1 - int(huffman_charat) - q)
            # e是输出向量 是将目标单词分类的向量 用来更新xw
            e += grad * node.huffman_vector
            # 更新node的内部哈夫曼向量
            node.huffman_vector += grad * input_vector
            # norm：可以为l1、l2或max，默认为l2
            #
            # 若为l1时，样本各个特征值除以各个特征值的绝对值之和
            #
            # 若为l2时，样本各个特征值除以各个特征值的平方之和
            # In [8]: from sklearn import preprocessing
            #    ...: X = [[ 1., -1., 2.],
            #              [ 2., 0., 0.],
            #              [ 0., 1., -1.]]
            #    ...: normalizer = preprocessing.Normalizer().fit(X)#fit does nothing
            #    ...: normalizer
            #    ...:
            # Out[8]: Normalizer(copy=True, norm='l2')
            #
            # In [9]: normalizer.transform(X)
            # Out[9]:
            # array([[ 0.40824829, -0.40824829,  0.81649658],
            #        [ 1.        ,  0.        ,  0.        ],
            #        [ 0.        ,  0.70710678, -0.70710678]])
            node.huffman_vector = preprocessing.normalize(node.huffman_vector)
            # 0 向右边 1 向左边
            if huffman_charat == '0':
                node = node.right
            else:
                node = node.left
        return e

    def __Sigmoid(self, value):
        return 1 / (1 + np.math.exp(-value))

    def generate_vocabulary(self, vocabulary):
        frequency_counts = 0
        for key in vocabulary.keys():
            frequency_counts += vocabulary[key]

        vocabulary_word = dict()
        index = 0
        for item in vocabulary:
            temp_dict = dict(
                word=item,
                freq=vocabulary[item],
                possibility=vocabulary[item] / frequency_counts,
                huffman_vector=np.random.random([1, self.vector_length]),
                huffman_code=None,
                index=index
            )
            index += 1
            vocabulary_word[item] = temp_dict
        self.vocabulary = vocabulary_word
        return vocabulary_word


# 定义哈夫曼节点对象
class huffman_node():
    def __init__(self,
                 # 当前叶子节点，保存的单词的值，比如说单词'this'
                 value,
                 # 哈夫曼节点 当时叶子节点的时候 会有当前词的频率
                 huffman_possibility,
                 # 默认不是叶子节点
                 is_leaf=False,
                 ):
        # 定义哈夫曼当前节点是从上个节点的左拐还是右拐
        self.huffman_direction = None
        # 定义哈夫曼节点对象中的向量θ
        self.huffman_vector = None
        # 定义哈夫曼树的左子树和右子树
        self.left = None
        self.right = None
        # 定义是否是叶子节点
        self.is_leaf = is_leaf
        # 定义哈夫曼节点对象中的哈夫曼编码
        self.huffman_code = ''
        self.value = value
        self.huffman_possibility = huffman_possibility


class huffman(object):
    '''
    huffman 有哈夫曼编码、向量、单词、左边、右边
    '''

    def __init__(self,
                 vocabulary,
                 vector_length=15000,

                 ):
        self.root = None
        self.vector_length = vector_length
        self.vocabulary = list(vocabulary.values())
        node_list = [huffman_node(x['word'], x['possibility'], True) for x in self.vocabulary]
        # 组装一个哈夫曼树形结构
        self.build_huffman_tree(node_list)
        self.generate_huffman_code(self.root, vocabulary)

    # 组装哈夫曼树
    # 现在的node_list只是按照possibility来排序的一个顺序表，下面是将这个node_list加上节点
    # 思路：每两个node作为一个节点
    def build_huffman_tree(self, node_list):
        current_node_list = []
        node_list.sort(key=lambda x: x.huffman_possibility, reverse=True)

        while len(node_list) != 1:
            # 最小的
            minimal = node_list[-1]
            # 次小的
            minor = node_list[-2]
            root = self.merge_left_right_and_root(minimal, minor)
            # 将已经使用过的树的部分删除掉
            node_list.pop(-1)
            node_list.pop(-1)
            node_list.append(root)
            # 按照节点的possibility进行重新排序
            node_list.sort(key=lambda x: x.huffman_possibility, reverse=True)
        # 倒数第一个 也仅有一个是树
        self.root = node_list[-1]

    # 构造出当前这一对叶子节点与根节点的数对
    def merge_left_right_and_root(self, node1, node2):
        root_possibility = node1.huffman_possibility + node2.huffman_possibility
        root = huffman_node(None, root_possibility, False)
        # 初始化θ
        root.huffman_vector = np.zeros([1, self.vector_length])
        if node1.huffman_possibility > node2.huffman_possibility:
            # 大的都放左边
            root.left = node1
            root.right = node2
        else:
            root.left = node2
            root.right = node1
        return root

    # 产生哈夫曼编码 利用递归
    def generate_huffman_code(self, root, vocabulary):

        # 到叶子节点就结束了
        if root.is_leaf is True:
            vocabulary[root.value]['huffman_code'] = root.huffman_code
            return
        else:
            # 根节点
            # TODO 这里用到的就是 哈夫曼编码
            code = root.huffman_code
            # 如果是最上面的根节点 那么它的huffman是code
            left_node = root.left
            # 左边是 1 右边是 0
            if left_node != None:
                left_code = code + '1'
                left_node.huffman_code = left_code
                # 然后再判定一下 如果左 或者 右 是叶子 那么久赋值
                self.generate_huffman_code(left_node, vocabulary)
            right_node = root.right
            if right_node != None:
                right_code = code + '0'
                right_node.huffman_code = right_code
                # 只有叶子节点在字典表里面才能找得到
                self.generate_huffman_code(right_node, vocabulary)


'''
用来保存最后结果的对象
'''


class vocabulary_word(object):
    def __init__(self,
                 word,
                 vector,
                 index,
                 count):
        self.word = word
        self.vector = vector
        self.index = index
        self.count = count


if __name__ == '__main__':
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
    input = '你是王八蛋' \
            '我是王九蛋'
    word2vec = word2vector(vector_length=100)
    word2vec.train(data)
    vocabulary = word2vec.vocabulary


    def cal_simi(data, key1, key2):
        huffman1 = data[key1]['huffman_vector']
        huffman1 = preprocessing.normalize(huffman1)
        huffman2 = data[key2]['huffman_vector']
        huffman2 = preprocessing.normalize(huffman2)
        e = huffman1.dot(huffman2.T)[0][0]
        return e


    keys = list(vocabulary.keys())
    possibility_list = []
    for key in keys:
        every = huffman_node(
            key, cal_simi(vocabulary, '漂亮', key))
        possibility_list.append(every)

    possibility_list.sort(key=lambda x: x.huffman_possibility, reverse=True)

    for obj in possibility_list:
        print(obj.value, '\t', obj.huffman_possibility)