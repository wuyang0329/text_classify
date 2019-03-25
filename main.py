from data_util import *
from LSTM import LSTM_model
from keras.callbacks import History,TensorBoard
from keras import backend as K
from SVM import *

from sklearn.externals import joblib

file_path = 'sum_bak.txt'
word2vec_path = 'cb_128_classify_new.vector'
word2index_path = 'word2index.txt'
log_filepath = './log'
BATCH_SIZE = 64

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

num_words,embedding_matrix,Xtrain,ytrain,Xtest,ytest = data_preprocess(file_path,word2vec_path,word2index_path)

model = LSTM_model(num_words,embedding_matrix).lstm()

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=[f1, "accuracy"])
## 网络训练

tbCallBack = TensorBoard(log_dir='./log',histogram_freq=1,write_graph=True,write_images=True)

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=20, validation_data=(Xtest, ytest),  callbacks=[tbCallBack])
#保存训练好的模型用于后面的SVM预测
model.save('model.h5')

#训练SVM模型
svc(traindata=Xtrain,trainlabel=ytrain,testdata=Xtest,testlabel=ytest,model_path='model.h5')


#预测新数据
feature_test = get_feature_vec(Xtest,'model.h5')

RF=joblib.load('SVM_0320.model')
# #使用模型进行预测
#
print("Predict---------------->")
result=RF.predict(feature_test)
#
print("result:\n")
print(result)
