# final-1-kelseywang99
final-1-kelseywang99 created by GitHub Classroom


# README
实验中，

第一问采用了bert、naive_LSTM两种模型

第二问采用了word2vec作为embedding（用train、test训练得到）的LSTM模型（w2v_LSTM）

第三问采用了word2vec作为embedding（用train、test、unlabled train训练得到）的LSTM及self training方法的模型（self_training）

## all_model_train.sh
训练所有模型的sh文件，其中$1、$2、$3分别为train、test、unlabeled train的csv路径，例如，$1 = ../train.csv， $2 = ../test.csv，$3 = ../train_unlabeled.csv

模型训练得到的param（bert.param 、naive_LSTM.param、w2v_LSTM.param、self_training.param）以及w2v的训练结果（第二问为w2v.model、第三问为w2v_all.model）保存在param文件夹中

python3 Q1_Bert_train.py $1 param/bert.param 

python3 Q1_Naive_LSTM_train.py $1 $2 param/naive_LSTM.param
 
python3 Q2_w2v_LSTM_train.py $1 $2 param/w2v_LSTM.param param/w2v.model  

python3 Q3_LSTM_Self_Training.py $1 $3 $2 param/self_training.param param/w2v_all.model

用到的包：
- Bert模型在训练中用到了transformers的包得到的（用到了BertConfig, BertForSequenceClassification, BertTokenizer, AdamW）
- 非Bert模型都用到了jieba分词
- w2v模型用到了gensim.models中的word2vec

##### 从训练的结果来看，bert的验证集准确率最高，因此kaggle上传的是由bert模型训练得到的结果，后面的train.sh、test.sh是根据bert模型写的；github上也只上传了bert模型的参数、test的预测结果


## train.sh
python3 Q1_Bert_train.py $1 bert.param

$1: train.csv是数据集

bert.param是模型参数保存的地方

Bert模型在训练中用到了transformers的包得到的（用到了BertConfig, BertForSequenceClassification, BertTokenizer, AdamW）

## test.sh 
python3 Q1_Bert_test.py $1 bert.param $2 

$1 test.csv是数据集

$2 prediction/bert_predict.csv是预测test data后的结果保存地址

bert.param是模型时训练得到的参数保存的地方

Bert模型在训练中用到了transformers的包得到的（用到了BertConfig, BertForSequenceClassification, BertTokenizer）

