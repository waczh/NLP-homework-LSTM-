import torch
from vocab_cnews import Vocab
from torch import nn
import numpy as np


def read_pretrained_wordvec(path, vocab:Vocab, word_dim):
    """
    给vocab中的每个词分配词向量,如果有预先传入的训练好的词向量,则提取出来 Assign word vectors to each word in the vocab, if there are pre-passed trained word vectors, they are extracted
    path: 词向量存储路径
    vocab: 词典
    word_dim: 词向量的维度
    返回值是词典（按照序号）对应的词向量 The return value is the word vector corresponding to the dictionary (by ordinal number)
    """
    vecs = np.random.normal(0.0, 0.9, [len(vocab), word_dim]) # 先随机给词典中的每个词分一个随机词向量
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split()
            if line[0] in vocab.vocab:  # 在词典里则提取出来，存到序号对应的那一行去
                vecs[vocab.word2seq(line[0])] = np.asarray(line[1:], dtype='float32')
    return vecs


class MyLSTM(nn.Module):
    def __init__(self, vecs, vocab_size, word_dim, num_layer, hidden_size, label_num) -> None:
        super(MyLSTM, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(vecs).float()) # 原来没加.float()导致mac上后面出现类型错误
        self.embedding_layer.weight.requires_grad = False
        
        self.rnn = nn.LSTM(word_dim, hidden_size, num_layer)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),    #
            nn.Linear(hidden_size, label_num)
        )

    def forward(self, X):

        X = X.permute(1, 0)

        X = self.embedding_layer(X)

        outs, _ = self.rnn(X)
        # LSTM的输出中的最后一个cell的输出喂给全连接层做预测ns
        logits = self.fc(outs[-1])

        return logits
