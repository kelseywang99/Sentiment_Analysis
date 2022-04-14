#!/usr/bin/env bash

python3 Q1_Bert_test.py $1 bert.param $2 

# $1 test.csv 是数据集

# $2 prediction/bert_predict.csv 是预测test data后的结果保存地址

# bert.param是模型训练保存的参数