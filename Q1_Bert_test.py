#!/usr/bin/env python
# coding: utf-8

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

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW

def load_data(data, label=True):
    # 带标签的train
    if label:
        # 因为csv中用1、2代表负面、正面，需要转换为0、1
        return list(data['comment']), [y-1 for y in data['value']]
    # 不带标签的train、test
    return list(data['comment'])


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


# 用index替换word来表示每个句子，并将句子统一成一个长度
def convert_text_to_ids(tokenizer, text, max_len=20):
    # 如果text是str（一句话，不是list）
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, truncation=True, add_special_tokens=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    # 如果text是list
    else:
        input_ids = []
        token_type_ids = []
        # 将text中的每一个sentence转化
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, truncation=True, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    return input_ids, token_type_ids


# padding（虽然covert_text_to_ids限制了max length，但是bert tokenizer不会自动padding）
def seq_padding(tokenizer, sentences):
    # 把pad加入到id中
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    # 只有一句话，不用pad
    if len(sentences) <= 1:
        return torch.tensor(sentences)
    
    # sentences有多句话
    # 获取每个句子的长度
    L = [len(sen) for sen in sentences]
    # 获取最长句子的长度
    ML = max(L)
    # pad到sentences中最长的那句话的长度，不够长的用pad的id补充
    sentences = torch.Tensor([sen + [pad_id] * (ML - len(sen)) if len(sen) < ML else sen for sen in sentences])
    
    return sentences


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            # 将inputs list中的每个sentence转化为用ids表示
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, inputs)
            # 将sentence进行padding
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            # 转化为LongTensor     
            input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
            input_ids, token_type_ids = input_ids.to(device), token_type_ids.to(device)

            # 得到output label
            output = model(input_ids=input_ids, token_type_ids=token_type_ids)
            y_pred_prob = output[0]
            y_pred_labels = y_pred_prob.argmax(dim=1).squeeze()

            # 正面、负面label分别为2、1，而模型输出是1、0，需要+1处理
            ret_temp = [y+1 for y in y_pred_labels.int().tolist()]
            ret_output += ret_temp
    
    return ret_output


if __name__ == '__main__':
    test_file = sys.argv[1]
    # ../test.csv
    model_dir = sys.argv[2]
    # param/bert.param
    save_dir = sys.argv[3]
    # prediction/bert_predict.csv

    batch_size =128
    epoch = 10
    lr = 1e-5
    weight_decay = 1e-2
    device = torch.device("cuda")

    test_data = pd.read_csv(test_file, index_col=0)
    test_x = load_data(test_data)
    test_dataset = CommentDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)

    config = BertConfig.from_pretrained("bert-base-chinese", num_labels=2)
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", config=config)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = torch.load(model_dir)
    outputs = testing(batch_size, test_loader, model, device)

    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"value":outputs})
    tmp.to_csv(save_dir, index=False)