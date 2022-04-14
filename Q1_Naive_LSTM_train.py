#!/usr/bin/env python
# coding: utf-8

# # Q1. Naive RNN
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import torch
from torch import nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import jieba

# ### 1、Data
# Dataset

def load_data(data, label=True):
    # 带标签的train
    if label:
        # 因为csv中用1、2代表负面、正面，需要转换为0、1
        return list(data['comment']), [y-1 for y in data['value']]
    # 不带标签的train、test
    return list(data['comment'])

def tokenizer(x):
    # 对sentences的列表中的每一个sentence采用jieba分词
    text_list = [list(jieba.cut(comment)) for comment in x]
    return text_list

class CommentDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        # 不带标签的train、test
        if self.label is None: 
            return self.data[idx]
        # 带标签的train
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


# Data Preprocess
class Preprocess():
    def __init__(self, sentences, sen_len, embedding_dim, min=5):
        '''
        sentences: sentence的list
        sen_len: 每个sentence要统一的长度
        embedding_dim: sentence每个词的维度
        min: 每个词在所有的sentence中要出现出少次才被作为input计算
        '''
        self.sentences = sentences
        self.sen_len = sen_len
        # 字典，将word映射到index
        self.word2idx = {}
        # 统计每个word在所有sentence中出现的次数
        self.count = {}

    
    def make_embedding(self, min=5):
        # 统计sentence中的每个word都出现了多少次
        for s in self.sentences:
            for word in s:
              self.count[word] = self.count.get(word, 0) + 1
        # 只保留出现次数 > min 的词
        self.count = {word:value for word, value in self.count.items() if value > min}
        
        # 给所有词进行编号，一个词对应一个index
        for word in self.count:
          self.word2idx[word] = len(self.word2idx)
        # 将<pad> <unk>也进行编号
        # <pad>用于补足不够sen_len长的句子， <unk>补足不知道的词（eg. 出现次数不够min次）
        self.word2idx["<PAD>"] = len(self.word2idx)
        self.word2idx["<UNK>"] = len(self.word2idx)

        # 用troch.nn.Embedding进行随机Embedding
        self.embedding_matrix = torch.nn.Embedding(len(self.word2idx), embedding_dim)
        
        print("total words: {}".format(len(self.word2idx)))

        # 返回embedding matrix
        return self.embedding_matrix

    # 将所有句子统一成sen_len长
    def pad_sequence(self, sentence):
        # > sen_len的句子截断
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        # < sen_len的句子用<pad>补齐
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    # 用index替换word来表示每个句子，并将句子统一成一个长度
    def sentence_word2idx(self, x):
        sentence_list = []
        # 对每一个sentence
        for i, sen in enumerate(x):
            # 用于记录该sentece的index表达
            sentence_idx = []
            for word in sen:
                # 如果在统计的word中(出现次数>=min)，通过查阅word2idx字典来转换
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                # 如果不在统计的word中(出现次数<min)，用<unk>的index替代
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将句子统一成一个长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        # 返回处理过的sentence_index
        return torch.LongTensor(sentence_list)
  
    # 把y的labels转换为longtensor
    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


# ### 2、Model
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        '''
        embedding: input的embedding
        embedding_dim: embedding的维度
        hidden_dim: lstm hidden_dim
        num_layers: lstm num_layers
        dropout: classifier全连接层的dropout
        '''
        super(LSTM_Net, self).__init__()

        self.embedding = embedding
        # embedding固定，不参与训练，因此不参与梯度下降
        self.embedding.weight.requires_grad = False
        self.embedding_dim = embedding
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # output全连接层
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 64),
                                         nn.Dropout(dropout),
                                         nn.Linear(64,1),
                                         nn.Sigmoid() )
        
    def forward(self, inputs):
        # embedding层
        inputs = self.embedding(inputs)
        # LSTM层
        x, _ = self.lstm(inputs, None)
        # 取LSTM最后一层的hidden state
        x = x[:, -1, :] 
        # 分类层
        x = self.classifier(x)
        return x

# 预测正确个数统计
def evaluation(outputs, labels):
    # >= 0.5 为正面，< 0.5 为负面
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    # 计算分类正确的个数
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


# ### 3、 Train
def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    '''
    model_dir: 模型参数保存地址
    train: train dataset loader
    valid: valid dataset loader
    model: embedding + lstm + calssifier的model
    device: cuda
    '''
    # 记录train和valid 的loss和acc
    train_loss = []
    valid_loss = []
    train_accs = []
    valid_accs = []

    # 总共的参数（含embedding）
    total = sum(p.numel() for p in model.parameters())
    # 可以训练的参数（不含embedding，含lstm+classifier）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    t_batch = len(train) 
    v_batch = len(valid) 

    # 二分类，采用binary cross entropy loss
    criterion = nn.BCELoss()
    # 采用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr) 

    model.train()
    
    best_acc = 0

    for epoch in range(n_epoch):
        # 记录本个epoch的total loss和acc
        total_loss, total_acc = 0, 0
        # training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) 
            # 因为要之后要用criterion，所以要转换为float
            labels = labels.to(device, dtype=torch.float)
            # 每个batch开始时要清0
            optimizer.zero_grad()
            outputs = model(inputs)
            # 因为要之后要用criterion，所以要去掉最外面的 dimension
            outputs = outputs.squeeze()
            # 计算training loss
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            # 统计模型预测正确的个数
            correct = evaluation(outputs, labels)
            # 更新本个epoch的total acc、total loss
            total_acc += (correct / batch_size)
            total_loss += loss.item()
        
        print('\n[ Epoch{}: ] \nTrain | Loss:{:.5f} Acc: {:.3f}'.format(epoch+1, total_loss/t_batch, total_acc/t_batch*100))
        
        # 记录该epoch的training平均loss、acc
        train_loss.append(total_loss/t_batch)
        train_accs.append(total_acc/t_batch)

        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) 
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                # validation loss、 acc
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            # 记录该epoch的validation平均loss、acc
            valid_loss.append(total_loss/v_batch)
            valid_accs.append(total_acc/v_batch)

            if total_acc > best_acc:
                # 如果 validation 的结果比之前最好的结果好，则保留本个epoch的参数
                best_acc = total_acc
                torch.save(model, "{}".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')

        # 准备进入下一个epoch，调整model模式为train
        model.train()
    
    # 返回每个epoch的train、valid的平均loss和acc的list
    return train_loss, train_accs, valid_loss, valid_accs


# ### 4、Run
if __name__ == '__main__':
    train_file = sys.argv[1]
    # ../train.csv
    test_file = sys.argv[2]
    # ../test.csv
    model_dir = sys.argv[3]
    # param/naive_LSTM.param

    sen_len = 20
    batch_size = 128
    epoch = 10
    lr = 0.001
    embedding_dim = 250

    # 读取数据，train只用有label的数据
    train_data = pd.read_csv(train_file,index_col=0)
    test_data = pd.read_csv(test_file,index_col=0)

    train_x, train_y = load_data(train_data)
    test_x = load_data(test_data, False)

    # tokenize
    train_x = tokenizer(train_x)
    test_x = tokenizer(test_x)

    # 取有label的train和test的所有sentence，进行预处理
    preprocess = Preprocess(train_x+test_x, sen_len, embedding_dim)
    embedding = preprocess.make_embedding()

    # 将train和test的输入转化为用id list表示的sentence的list
    train_x = preprocess.sentence_word2idx(train_x)
    y = preprocess.labels_to_tensor(train_y)

    # 把training data取90%为train，10%为valid
    X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

    train_dataset = CommentDataset(X=X_train, y=y_train)
    val_dataset = CommentDataset(X=X_val, y=y_val)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 8)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)

    # 初始化model
    model = LSTM_Net(embedding, embedding_dim, hidden_dim=150, num_layers=1, dropout=0.3)
    device = torch.device("cuda" )
    model = model.to(device)

    # 开始训练
    train_loss, train_accs, valid_loss, valid_accs = training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

    # 绘制训练过程的曲线图
    fig = plt.figure()
    plt.plot(list(range(1, 1+epoch)), train_accs, label='train_acc')
    plt.plot(list(range(1, 1+epoch)), valid_accs, label='valid_acc')
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.plot(list(range(1, 1+epoch)), train_loss, label='train_loss')
    plt.plot(list(range(1, 1+epoch)), valid_loss, label='valid_loss')
    plt.legend()
    plt.show()
