#!/usr/bin/env python
# coding: utf-8

# # Q3. LSTM_Self_Training
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

import torch
from torch import nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import jieba
from gensim.models import word2vec


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
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 读取之前训练好的 word to vec 模型
        self.embedding = word2vec.Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size


    def add_embedding(self, word):
        # 把 "<PAD>""<UNK>"加入 embedding
        vector = torch.empty(1, self.embedding_dim)
        # 随机初始化
        torch.nn.init.uniform_(vector)
        # 加入到word2idx
        self.word2idx[word] = len(self.word2idx)
        # 加入到idx2word
        self.idx2word.append(word)
        # 加入到embeeding_matrix
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    # 制作embedding matrix、word2idx的字典、idx2word的list
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 读取之前训练好的 word to vec embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.vocab):
            # 加入到word2idx的字典中，key=wrod，value=id
            self.word2idx[word] = len(self.word2idx)
            # 加入到idx2word的list中
            self.idx2word.append(word)
            # 加入到embedding_matrix list中，id位置上是id的word的特征向量
            self.embedding_matrix.append(self.embedding[word])
        # embedding matrix转化为tensor
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 加入"<PAD>" "<UNK>" 到embedding中
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")

        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix


    # 将句子变化为统一长度  
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
    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                # word在embedding，用id替换
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                # word不在embedding中的用<unk>的id补充
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将句子统一成一个长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        # 返回处理过的sentence_index
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
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
        # 制作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # embedding固定，不参与训练，因此不参与梯度下降
        self.embedding.weight.requires_grad = False 
        self.embedding_dim = embedding.size(1)
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


# ### 3、Train
# 根据现有模型，得到预测置信度高的无标签训练数据
def judge_x_nolabel(train_unlabeled_x_loader, model, device, batch_size, threshold=0.8):
    model.eval()
    ret_inputs = []
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(train_unlabeled_x_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # 如果outputs的比threshold大，或者比1-threshold小，则置信度高，加入到返回的list中
            temp_inputs = [inputs[i].tolist() for i in range(len(outputs)) if (outputs[i] > threshold or outputs[i] < 1 - threshold)]
            if len(temp_inputs) == 0:
                continue
            ret_inputs += temp_inputs
            temp_inputs = torch.LongTensor(temp_inputs)

            # 得到新加入的高置信度的data的预测label
            outputs = model(temp_inputs.to(device, dtype=torch.long))
            if len(outputs) == 1:
                y = 1 if outputs[0]>=0.5 else 0
                ret_put.append(y)
                continue
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1
            outputs[outputs<0.5] = 0
            ret_output += outputs.int().tolist()
    
    # 输出一共有多少高置信度的label
    print(len(ret_inputs))
    # 转化为longtensor
    ret_inputs = torch.LongTensor(ret_inputs)
    
    return ret_inputs, ret_output

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

# def testing(test_loader, model):
#     model.eval()
#     # 记录output的list
#     ret_output = []
#     with torch.no_grad():
#         for i, inputs in enumerate(test_loader):
#             inputs = inputs.to(device, dtype=torch.long)
#             outputs = model(inputs)
#             outputs = outputs.squeeze()
#             # >= 0.5 为正面，label = 2
#             outputs[outputs>=0.5] = 2 
#             # < 0.5 为负面，label = 1
#             outputs[outputs<0.5] = 1 
#             ret_output += outputs.int().tolist()
    
#     return ret_output

# ### 4、Run
if __name__ == '__main__':
    train_file = sys.argv[1]
    # ../train.csv
    unlabeled_train_file = sys.argv[2]
    #  ../train_unlabeled.csv
    test_file = sys.argv[3]
    # ../test.csv
    model_dir = sys.argv[4]
    # param/self_training.param
    w2v_path = sys.argv[5]
    # param/w2v_all.model
    
    # save_dir = sys.argv[6]
    # # prediction/self_training_predict.csv
    # train_file = '../train.csv'
    # unlabeled_train_file = '../train_unlabeled.csv'
    # test_file = '../test.csv'
    # model_dir = 'param/self_training.param'
    # w2v_path = 'param/w2v_all.model'
    # save_dir = 'prediction/self_training_predict.csv'

    sen_len = 20
    batch_size = 128
    epoch = 10
    lr = 0.001
    device = torch.device("cuda")
    threshold = 0.9

    # 读取数据
    train_data = pd.read_csv(train_file,index_col=0)
    train_unlabeled_data = pd.read_csv(unlabeled_train_file,index_col=0)
    test_data = pd.read_csv(test_file,index_col=0)

    train_x, train_y = load_data(train_data)
    train_unlabeled_x = load_data(train_unlabeled_data, False)
    test_x = load_data(test_data, False)

    # tokenize
    train_x = tokenizer(train_x)
    train_unlabeled_x =tokenizer(train_unlabeled_x)
    test_x = tokenizer(test_x)

    # 预处理带标签的train data、制作embedding
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x0 = preprocess.sentence_word2idx()
    y0 = preprocess.labels_to_tensor(train_y)

    # 把带标签的training data取90%为train，10%为valid
    X_train, X_val, y_train, y_val = train_x0[:180000], train_x0[180000:], y0[:180000], y0[180000:]

    train_dataset = CommentDataset(X=X_train, y=y_train)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 8)


    val_dataset = CommentDataset(X=X_val, y=y_val)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)

    # 预处理不带标签的train data
    preprocess = Preprocess(train_unlabeled_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    X_train_unlabeled = preprocess.sentence_word2idx()

    train_unlabeled_dataset = CommentDataset(X=X_train_unlabeled, y=None)

    train_unlabeld_x_loader = torch.utils.data.DataLoader(dataset = train_unlabeled_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

    # 初始化model
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.3)
    device = torch.device("cuda")
    model = model.to(device)

    # 训练
    train_loss, train_accs, valid_loss, valid_accs = training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)


    # #### label unlabeled data
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.3)
    device = torch.device("cuda")
    model = torch.load(model_dir)
    # 得到置信度高的unlabel train data
    train_unlabeled_goodx, train_unlabeled_goody = judge_x_nolabel(train_unlabeld_x_loader, model, device, batch_size, threshold)
    train_unlabeled_goodx1 = train_unlabeled_goodx
    train_unlabeled_goody1 = torch.LongTensor(train_unlabeled_goody)

    # 将置信度高的unlabel train data与有标签的train data拼接
    X =  torch.cat((X_train, train_unlabeled_goodx1), 0)
    y =  torch.cat((y_train, train_unlabeled_goody1), 0)
    train_dataset = CommentDataset(X=X, y=y)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 8)


    # #### train labeled + unlabeled data
    # 初始化model
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.3)
    device = torch.device("cuda" )
    model = model.to(device) 
    # 训练
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


    # # ## Test
    # # 对test data进行预处理，制作embedding
    # preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    # embedding = preprocess.make_embedding(load=True)
    # test_x = preprocess.sentence_word2idx()

    # test_dataset = CommentDataset(X=test_x, y=None)

    # test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
    #                                             batch_size = batch_size,
    #                                             shuffle = False,
    #                                             num_workers = 8)
    # model = torch.load(model_dir)
    # outputs = testing(test_loader, model)

    # tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"value":outputs})
    # tmp.to_csv(save_dir, index=False)

