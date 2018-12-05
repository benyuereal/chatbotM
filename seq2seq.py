import random
import time

import jieba
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from word2vector import word2vector

'''
配置区域
'''
answer_path = "./static/answer"
question_path = "./static/question"
USE_CUDA = torch.cuda.is_available()
SOS_token = 2
EOS_token = 1
batch_size = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        print(self.embedding.weight)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        print(embedded)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        # 46:00开始看视频 TODO 注意ht是输入的每一个时刻的隐含变量 softmax计算输入权重概率分布 e是解码的时候上一时刻的隐藏状态和输入的时候的隐含状态
        attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        for i in range(seq_len):
            # ht * c
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, self.max_length)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N
        # 连接当前输入和上一个背景向量
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        # output = self.out(torch.cat((rnn_output, context), 1))
        return output, context, hidden, attn_weights


class seq2seq(nn.Module):

    def __init__(self, data=None):
        super(seq2seq, self).__init__()
        self.data = data
        self.vocabulary_word = None
        self.vocabulary_index = None
        self.enc_vec = []
        self.dec_vec = []
        self.max_epoches = 100000
        self.batch_index = 0
        self.GO_token = 2
        self.EOS_token = 1
        self.input_size = 14
        self.output_size = 15
        self.hidden_size = 100
        self.max_length = 15
        self.show_epoch = 100
        self.use_cuda = USE_CUDA
        self.model_path = "./model/"
        self.n_layers = 1
        self.dropout_p = 0.05
        self.beam_search = True
        self.top_k = 5
        self.alpha = 0.5
        # 初始化encoder和decoder
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = AttnDecoderRNN('general', self.hidden_size, self.output_size, self.n_layers, self.dropout_p,
                                      self.max_length)

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.criterion = nn.NLLLoss()

    def next(self, batch_size, eos_token=1, go_token=2, shuffle=False):
        inputs = []
        targets = []

        if shuffle:
            ind = random.choice(range(len(self.enc_vec)))
            enc = [self.enc_vec[ind]]
            dec = [self.dec_vec[ind]]
        else:
            if self.batch_index + batch_size >= len(self.enc_vec):
                enc = self.enc_vec[self.batch_index:]
                dec = self.dec_vec[self.batch_index:]
                self.batch_index = 0
            else:
                enc = self.enc_vec[self.batch_index:self.batch_index + batch_size]
                dec = self.dec_vec[self.batch_index:self.batch_index + batch_size]
                self.batch_index += batch_size
        for index in range(len(enc)):
            enc = enc[0][:self.max_length] if len(enc[0]) > self.max_length else enc[0]
            dec = dec[0][:self.max_length] if len(dec[0]) > self.max_length else dec[0]

            enc = [int(i) for i in enc]
            dec = [int(i) for i in dec]
            dec.append(eos_token)

            inputs.append(enc)
            targets.append(dec)

        inputs = Variable(torch.LongTensor(inputs)).transpose(1, 0).contiguous()
        targets = Variable(torch.LongTensor(targets)).transpose(1, 0).contiguous()
        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()
        return inputs, targets

    def step(self, input_variable, target_variable, max_length):
        teacher_forcing_ratio = 0.1
        clip = 5.0
        loss = 0  # Added onto for each word

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_hidden = self.encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_context = Variable(torch.zeros(1, self.decoder.hidden_size))
        decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            decoder_context = decoder_context.cuda()

        decoder_outputs = []
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        use_teacher_forcing = True
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                                  decoder_context,
                                                                                                  decoder_hidden,
                                                                                                  encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output.unsqueeze(0))
        else:
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                                  decoder_context,
                                                                                                  decoder_hidden,
                                                                                                  encoder_outputs)
                loss += self.criterion(decoder_output, target_variable[di])
                decoder_outputs.append(decoder_output.unsqueeze(0))
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                if USE_CUDA: decoder_input = decoder_input.cuda()
                if ni == EOS_token: break
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        decoder_outputs = torch.cat(decoder_outputs, 0)
        return loss.data[0] / target_length, decoder_outputs

    '''
    训练开始 首先是加载字典 然后是加载数据索引
    '''

    def train(self):
        self.get_vocabulary()
        self.load_data()
        try:
            self.load_state_dict(torch.load(self.model_path + 'params.pkl'))
        except Exception as e:
            print(e)
            print("No model!")
        loss_track = []

        for epoch in range(self.max_epoches):
            start = time.time()
            inputs, targets = self.next(batch_size, shuffle=False)
            loss, logits = self.step(inputs, targets, self.max_length)
            loss_track.append(loss)
            _, v = torch.topk(logits, 1)
            pre = v.cpu().data.numpy().T.tolist()[0][0]
            tar = targets.cpu().data.numpy().T.tolist()[0]
            stop = time.time()
            if epoch % self.show_epoch == 0:
                print("-" * 50)
                print("epoch:", epoch)
                print("    loss:", loss)
                print("    target:%s\n    output:%s" % (tar, pre))
                print("    per-time:", (stop - start))
                torch.save(self.state_dict(), self.model_path + 'params.pkl')

    def load_data(self):
        vocabulary = self.vocabulary_word
        enc_vec = []
        dec_vec = []
        with open(question_path) as enc:
            line = enc.readline()

            while line:
                index_list = []
                line = jieba.cut(line, cut_all=True)
                for word in line:
                    if word in vocabulary.keys():
                        node = vocabulary[word]
                        index_list.append(node.index)
                enc_vec.append(index_list)
                line = enc.readline()

        with open(answer_path) as dec:
            line = dec.readline()
            while line:
                index_list = []
                line = jieba.cut(line, cut_all=True)
                for word in line:
                    if word in vocabulary.keys():
                        node = vocabulary[word]
                        index_list.append(node.index)
                dec_vec.append(index_list)
                line = dec.readline()

        self.dec_vec = dec_vec
        self.enc_vec = enc_vec

    def get_data(self):
        data = [
        ]

        question = open(question_path)  # 返回一个文件对象
        answer = open(answer_path)  # 返回一个文件对象
        line_answer = answer.readline()  # 调用文件的 readline()方法
        line_question = question.readline()
        while line_question and line_answer:
            line_question = question.readline()
            line_answer = answer.readline()
            data.append(line_question)
            data.append(line_answer)
        answer.close()
        question.close()
        return data

    '''
    获取字典数据
    '''

    def get_vocabulary(self):
        try:
            vocabulary_word = torch.load(self.model_path + 'vocabulary_word.pkl')
            vocabulary_index = torch.load(self.model_path + 'vocabulary_index.pkl')
            self.vocabulary_word = vocabulary_word
            self.vocabulary_index = vocabulary_index
        except Exception as e:
            data = self.get_data()
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
            torch.save(vocabulary_word, self.model_path + 'vocabulary_word.pkl')
            torch.save(vocabulary_index, self.model_path + 'vocabulary_index.pkl')


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
    seq2seq.train()
