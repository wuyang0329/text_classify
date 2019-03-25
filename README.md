## 项目简介
学习LSTM神经网络，循环神经网络在实际环境下的应用，提升实践能力，了解深度学习在自然语言处理方面的进展

## lstm+svm_for_text_classify
具有较强的对长难句，反问句，阴阳怪气句的判断能力，在在酒店评论测试集上达到97%的准确率  
采用双向LSTM网络提取句子特征，训练SVM分类器进行文本分类任务，还有Attention层
对输入数据进行dropout，模拟增大样本空间  
LSTM层与层之间进行dropout  
对LSTM网络权重，偏置进行l2正则，抗过拟合  

## word2vec:
项目使用的词向量：embedding_64.bin(1.5G)  
训练语料：百度百科800w条 20G+搜狐新闻400w条 12G+小说：90G左右  
模型参数：window=5 min_count=5 size=64  
下载链接：[百度网盘链接](https://pan.baidu.com/s/19bDbZsFzLggx7q9iFn83Nw)

## 文件功能介绍
./  
segment.py：对训练数据进行分词，用于训练词向量
data_util.py：数据处理函数
LSTM.py：定义lstm模型
SVM.py：svm分类器训练函数  
main.py：项目主文件  

## 推荐运行环境
python 3.6  
tensorflow-gpu 1.7  
gensim 3.4.0  
Ubuntu 64 Bit / windows10 64 Bit  

## 使用注意事项
1.文本TXT文件必须采用UTF-8编码格式，非UTF-8格式的，去记事本中另存为的时候选择UTF-8  
2.训练数据文件一行为一条评论，长度不限，可以有英文和标点（反正都会去除的），不要词性标注信息   
3.测试集比率根据样本数量自行调整，太大容易造成显存不够导致失败  
4.根据文件夹结构自行建立  
