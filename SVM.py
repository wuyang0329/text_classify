#coding:utf-8
from __future__ import print_function

from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib
from keras import backend as K
from data_util import get_feature_vec
import gensim
import nltk  #用来分词
import collections  #用来统计词频
import numpy as np


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



def svc(traindata, trainlabel, testdata, testlabel,model_path):
    feature_train = get_feature_vec(traindata,model_path)
    feature_test = get_feature_vec(testdata,model_path)
    print("Start training SVM...")
    svcClf = SVC(C=0.9, kernel="rbf", cache_size=3000)
    svcClf.fit(feature_train, trainlabel)
    pred_testlabel = svcClf.predict(feature_test)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("cnn-svm Accuracy:", accuracy)
    #保存模型
    joblib.dump(svcClf,'SVM_03.model')



# svc(feature_train[:5000,:],ytrain[:5000],feature_test,ytest)


# from data_util import sentences2vec
#
# x = "发现这家公园评价这家公园童年回忆小时候住记得小时候人气旺老爷爷老奶奶喜欢锻炼身体记得有颗树特别小朋友做秋千吃早饭怀念味道这家公园总体不错"
# Xtest = sentences2vec(x,word2index_path='word2index.txt')
# #加载模型
# RF=joblib.load('SVM.model')
#使用模型进行预测

# result=RF.predict(Xtest[0])
# print(result)

