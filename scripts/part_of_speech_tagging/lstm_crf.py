# encoding:utf-8

import sys
import re
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import Block, nn, rnn
# import mxnet.optimizer as optimizer

'''
训练、测试语料读取
'''
sents = open('./icwb2-data/training/msr_training.utf8').read()
#  sents = sents.decode('utf-8').strip()
sents = sents.strip()
#  sents = sents.split('\r\n') # 这个语料的换行符是\r\n
sents = sents.split('\n')

'''
输出语料中句子的个数
'''
print(len(sents))
print(sents[0])

sents = [re.split(' +', s) for s in sents]  # 词之间以空格隔开
sents = [[w for w in s if w] for s in sents]  # 去掉空字符串
np.random.shuffle(sents)  # 打乱语料，以便后面划分验证集

chars = {}  # 统计字表
for s in sents:
    for c in ''.join(s):
        if c in chars:
            chars[c] += 1
        else:
            chars[c] = 1

min_count = 2  # 过滤低频字
chars = {i: j for i, j in chars.items() if j >= min_count}  # 过滤低频字
id2char = {i + 1: j for i, j in enumerate(chars)}  # id到字的映射
char2id = {j: i for i, j in id2char.items()}  # 字到id的映射
print(len(id2char))
print(len(char2id))

id2tag = {0: 's', 1: 'b', 2: 'm', 3: 'e', 4: 'p'}  # 标签（sbme）与id之间的映射
tag2id = {j: i for i, j in id2tag.items()}
print(len(id2tag))
print(len(tag2id))

train_sents = sents[:-5000]  # 留下5000个句子做验证，剩下的都用来训练
valid_sents = sents[-5000:]
print(len(train_sents))
print(len(valid_sents))

'''
网络超参数或者
'''
vocab_size = len(id2char)  # 词表大小
embed_size = 100  # Embedding层参数
num_hiddens = 150  # LSTM隐藏层维度
num_layers = 2  # LSTM层数
batch_size = 128  # 一个batch个数

# lr = 0.5
# clipping_theta = 0.2
num_epochs = 1
# num_steps = 5
# drop_prob = 0.2
# eval_period = 100
bidirectional = True  # 双向LSTM
ignore_last_label = True  # 未用到


'''
定义训练数据生成器
'''
def train_generator():
    while True:
        X, Y = [], []
        for i, s in enumerate(train_sents):  # 遍历每个句子
            sx, sy = [], []
            for w in s:  # 遍历句子中的每个词
                sx.extend([char2id.get(c, 0) for c in w])  # 遍历词中的每个字
                if len(w) == 1:
                    sy.append(0)  # 单字词的标签
                elif len(w) == 2:
                    sy.extend([1, 3])  # 双字词的标签
                else:
                    sy.extend([1] + [2] * (len(w) - 2) + [3])  # 多于两字的词的标签
            X.append(sx)
            Y.append(sy)
            if len(X) == batch_size or i == len(train_sents) - 1:  # 如果达到一个batch
                maxlen = max([len(x) for x in X])  # 找出最大字数
                X = [x + [0] * (maxlen - len(x)) for x in X]  # 不足则补零
                Y = [y + [4] * (maxlen - len(y)) for y in Y]  # 不足则补第五个标签
                yield np.array(X), np.array(Y)
                X, Y = [], []

'''
定义测试数据生成器
'''
def test_generator():
    while True:
        X,Y = [],[]
        for i,s in enumerate(valid_sents):  # 遍历每个句子
            sx,sy = [],[]
            for w in s:  # 遍历句子中的每个词
                sx.extend([char2id.get(c, 0) for c in w])  # 遍历词中的每个字
                if len(w) == 1:
                    sy.append(0)  # 单字词的标签
                elif len(w) == 2:
                    sy.extend([1,3])  # 双字词的标签
                else:
                    sy.extend([1] + [2]*(len(w)-2) + [3])  # 多于两字的词的标签
            X.append(sx)
            Y.append(sy)
            if len(X) == batch_size or i == len(train_sents)-1:  # 如果达到一个batch
                maxlen = max([len(x) for x in X])  # 找出最大字数
                X = [x+[0]*(maxlen-len(x)) for x in X]  # 不足则补零
                Y = [y+[4]*(maxlen-len(y)) for y in Y]  # 不足则补第五个标签
                yield np.array(X), np.array(Y)
                X, Y = [],[]

'''                
log sum exp定义
'''
def log_sum_exp(vec, axis, keepdims=False):
    max_score = nd.max(vec).asscalar()
    return nd.log(nd.sum(nd.exp(vec - max_score), axis=axis, keepdims=keepdims)) + max_score

'''
定义一个求字典中最大值的函数
'''
def max_in_dict(d):
    key, value = list(d.items())[0]
    for i, j in list(d.items())[1:]:
        if j.asscalar() > value.asscalar():
            key, value = i, j
    return key, value.asscalar()

'''
网络结构定义
BiLSTM_CRF model
'''
class BiLSTM_CRF(Block):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tag2idx, num_layers, bidirectional,
                 ignore_last_label=False):
        super(BiLSTM_CRF, self).__init__()
        with self.name_scope():
            self.ignore_last_label = ignore_last_label
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.tagset_size = len(tag2idx) - 1
            self.num_layers = num_layers

            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

            # input shape: (sequence_length, batch_size, input_size)
            #
            # output shape: (sequence_length, batch_size, num_hidden),
            # If bidirectional is True, output shape: (sequence_length, batch_size, 2*num_hidden)
            #
            # recurrent state: (num_layers, batch_size, num_hidden)
            # If bidirectional is True, (2*num_layers, batch_size, num_hidden)
            # If input recurrent state is None, zeros are used as default begin states,
            # and the output recurrent state is omitted.
            self.lstm = rnn.LSTM(self.hidden_dim, num_layers=self.num_layers,
                                 bidirectional=bidirectional,
                                 input_size=self.embedding_dim)

            # Maps the output of the LSTM into tag space.
            self.hidden2tag = nn.Dense(
                self.tagset_size)

            # the crf layer.
            self.transitions = self.params.get('weight', shape=(self.tagset_size, self.tagset_size))

    def init_hidden(self, batch_size):
        return [nd.random.normal(shape=(self.num_layers, batch_size, self.hidden_dim)),
                nd.random.normal(shape=(self.num_layers, batch_size, self.hidden_dim))]

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)

    def _get_lstm_features(self, sentence, batch_size):
        self.hidden = self.begin_state(func=nd.zeros, batch_size=batch_size)

        # embeds shape: (sequence_length, batch_size, embedding_dim)
        embeds = self.embedding(sentence)

        # lstm_out shape: (sequence_length, batch_size, hidden_dim)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # lstm_out shape: (sequence_length*batch_size, hidden_dim*2)
        lstm_out = lstm_out.reshape((-1,
                                     self.hidden_dim * 2))

        # lstm_feats shape: (sequence_length*batch_size, tagset_size)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    '''
    viterbi算法
    '''
    def _viterbi_decode(self, nodes):
        key = list(range(len(nodes[0])))
        for i in key:
            key[i] = str(i)
        paths = dict(zip(key, nodes[0]))  # 初始化起始路径

        for l in range(1, len(nodes)):  # 遍历后面的节点
            path_old, paths = paths, {}

            for n, ns in enumerate(nodes[l]):  # 当前时刻的所有节点
                max_path, max_score = '', nd.array([-1e100])
                for p, ps in path_old.items():  # 截止至前一时刻的最优路径集合
                    score = ns + ps + (self.transitions.data())[int(p[-1])][n]  # 计算新分数
                    if score.asscalar() > max_score.asscalar():  # 如果新分数大于已有的最大分数
                        max_path, max_score = p + str(n), score  # 更新路径
                paths[max_path] = max_score  # 储存到当前时刻所有节点的最优路径
        return max_in_dict(paths)

    '''
    batch_size恒等于1
    '''
    def forward(self, inputs, batch_size):
        sequence_length_ = len(inputs)
        sequence_length = 0
        for j in range(sequence_length_):  # 函数目的是去掉padding
            if (inputs[j, 0].asscalar() <= 0):
                sequence_length = j
                break
            else:
                sequence_length = sequence_length_

        if (sequence_length == 0):
            print("sequence_length=0")
            print(inputs)
            return

        inputs = inputs[0:sequence_length, :]

        # Get the emission scores from the BiLSTM.
        # inputs.shape: (sequence_length, batch_size)
        lstm_feats = self._get_lstm_features(inputs, batch_size)

        '''
        目的：将[sequence_length, batch_size, tagset_size]维度转换为[batch_size, sequence_length, tagset_size]
        '''
        # outputs.shape: batch_size个(sequence_length, tagset_size)
        lstm_feats = nd.split(lstm_feats, num_outputs=batch_size, axis=0)
        # outputs.shape: (sequence_length, tagset_size)
        lstm_feats = nd.concat(*lstm_feats, dim=0).reshape(sequence_length, self.tagset_size)

        # Find the best path, given the features.
        tag_seq, score = self._viterbi_decode(lstm_feats)
        return tag_seq, score

    '''
    递归计算归一化因子
    要点：1、递归计算；2、用logsumexp避免溢出。
    '''
    def _forward_alg(self, feats, sequence_length):
        state = feats[:, 0]  # 初始状态
        output = state  # 如果sequence_length==1，output = state
        for i in range(1, sequence_length):
            state = nd.expand_dims(state, 2)  # (batch_size, tagset_size, 1)
            trans = nd.expand_dims(self.transitions.data(), 0)  # (1, tagset_size, tagset_size)
            output = log_sum_exp(state + trans, 1)
            output = output + feats[:, i]
            state = output
        return output

    """
    计算目标路径的相对概率（还没有归一化）
    要点：逐标签得分，加上转移概率得分。
    """
    def _score_sentence(self, feats, tags):
        point_score = nd.sum(nd.sum(feats * tags, 2), 1, keepdims=True)  # 逐标签得分
        if (feats.shape[1] == 1):  # 如果sequence_length==1，没有转移概率
            return point_score
        labels1 = nd.expand_dims(tags[:, :-1], 3)
        labels2 = nd.expand_dims(tags[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = nd.expand_dims(nd.expand_dims(self.transitions.data(), 0), 0)
        trans_score = nd.sum(nd.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    '''
    crf loss函数
    '''
    def neg_log_likelihood(self, inputs_, tags_, batch_size):
        loss_sum = nd.zeros((1, 1))
        '''
        避免pading的影响，一个batch的数据一行一行计算loss
        '''
        for i in range(batch_size):
            inputs = inputs_[:, i:i + 1]
            sequence_length_ = len(inputs)

            sequence_length = 0
            for j in range(sequence_length_):
                if (inputs[j, 0].asscalar() <= 0):
                    sequence_length = j
                    break
                else:
                    sequence_length = sequence_length_

            if (sequence_length == 0):
                print("sequence_length=0")
                print(inputs)
                batch_size = batch_size - 1
                continue

            inputs = inputs[0:sequence_length, :]
            tags = tags_[i:i + 1, 0:sequence_length, 0:4]

            # inputs.shape: (sequence_length, batch_size)
            lstm_feats = self._get_lstm_features(inputs, 1)

            # outputs.shape: batch_size个(sequence_length, tagset_size)
            lstm_feats = nd.split(lstm_feats, num_outputs=1, axis=0)

            # outputs.shape: (batch_size, sequence_length, tagset_size)
            lstm_feats = nd.concat(*lstm_feats, dim=0).reshape(-1, sequence_length, self.tagset_size)

            forward_score = self._forward_alg(lstm_feats, sequence_length)
            forward_score = log_sum_exp(forward_score, 1, keepdims=True)
            gold_score = self._score_sentence(lstm_feats, tags)
            if (forward_score.sum().asscalar() - gold_score.sum().asscalar() <= 0):
                print("loss < 0")
                print(forward_score)
                print(gold_score)
                print(inputs)
            loss_sum = (forward_score - gold_score) / batch_size + loss_sum
        return loss_sum

model = BiLSTM_CRF(vocab_size + 1, embed_size, num_hiddens, tag2id, num_layers, bidirectional, ignore_last_label)
print(model)
print(model.collect_params())
model.initialize(init.Xavier(magnitude=2.24))  # 参数初始化
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 1e-4})  # 定义优化器

'''
加载保存的权重
'''
import os
# symbol_file = os.path.join('triplet3-net') # 加载最新的权重
# model.load_params(filename=symbol_file)

'''
训练集、测试集初始化
'''
train_set = train_generator()
test_set = test_generator()

'''
计算测试集的loss
'''
def eval_rnn(data_source):
    l_sum = nd.array([0])
    n = 0
    for i in range(len(data_source)//batch_size):
        X, y = next(test_set)
        X, y = nd.array(X.T), mx.ndarray.one_hot(nd.array(y), 5)
        loss = model.neg_log_likelihood(X, y, batch_size)
        l_sum += loss.sum()
        n = n + 1
    return l_sum / n

'''
优化参数
'''
def train_rnn():
    for epoch in range(1, num_epochs + 1):
        for i in range(len(train_sents)//batch_size):
            X, y = next(train_set)
            batch_size_ = len(X)
            X, y = nd.array(X.T), mx.ndarray.one_hot(nd.array(y), 5)
            with autograd.record():
                loss = model.neg_log_likelihood(X, y, batch_size_)
            print(i)
            print(loss)
            print(loss.sum())
            loss.backward()
            optimizer.step(1)
            if ((i+1) % 50 == 0):
                val_l = eval_rnn(valid_sents)
                print('----------batch %d, valid loss %.2f' % (i, val_l.asscalar()))
        val_l = eval_rnn(valid_sents)
        print('--------------------epoch %d, valid loss %.2f' % (epoch, val_l.asscalar()))

train_rnn()

'''
保存最新的权重
'''
symbol_file = os.path.join('triplet3-net')  # 最新的权重
model.save_params(filename=symbol_file)
