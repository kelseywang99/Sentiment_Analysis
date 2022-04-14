#!/usr/bin/env bash

python3 Q1_Bert_train.py $1 param/bert.param & python3 Q1_Naive_LSTM_train.py $1 $2 param/naive_LSTM.param & python3 Q2_w2v_LSTM_train.py $1 $2 param/w2v_LSTM.param param/w2v.model  & python3 Q3_LSTM_Self_Training.py $1 $3 $2 param/self_training.param param/w2v_all.model

# $1 = ../train.csv
# $2 = ../test.csv 
# $3 = ../train_unlabeled.csv