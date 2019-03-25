from __future__ import print_function
from keras.layers.core import Activation, Dense, Dropout,Permute,Flatten,Reshape,RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
from keras import layers
from keras.models import *
from keras.preprocessing import sequence


# model = Sequential()
class LSTM_model():
	def __init__(self,num_words,embedding_matrix):
		#  网络构建
		self.EMBEDDING_SIZE = 128
		# 有关卷积的改在这
		self.filter_length1 = 3
		self.filter_length2 = 4
		self.filter_length3 = 5
		self.nb_filter = 64
		# 有关LSTM的参数
		self.HIDDEN_LAYER_SIZE = 128
		self.HIDDEN_LAYER_SIZE2 = 192
		# HIDDEN_LAYER_SIZE1 = 256
		self.BATCH_SIZE = 64
		self.NUM_EPOCHS = 10
		self.MAX_FEATURES =80000
		self.MAX_SENTENCE_LENGTH =300
		self.ALL_VECTOR = 160000

		self.num_words = num_words
		self.embedding_matrix = embedding_matrix
		self.log_filepath = './log'



	def lstm(self):

		embedding_layer = Embedding(self.num_words,
		                            self.EMBEDDING_SIZE,
		                            weights=[self.embedding_matrix],
		                            input_length=self.MAX_SENTENCE_LENGTH,
		                            trainable=False)
		main_input = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32', name='main_input')

		embedding = embedding_layer(main_input)
		# 加入卷积
		conv1 = layers.Conv1D(filters=self.nb_filter,kernel_size = self.filter_length1, padding='same', activation='relu',dilation_rate=1,name='conv1')(embedding)
		maxConv1 = layers.MaxPooling1D(pool_size=1,name='maxConv4')(conv1)
		conv2 = layers.Conv1D(filters=self.nb_filter,kernel_size= self.filter_length2, padding='same', activation='relu',dilation_rate=1,name='conv2')(embedding)
		maxConv2 = layers.MaxPooling1D(pool_size=1,name='maxConv2')(conv2)
		conv3 = layers.Conv1D(filters=self.nb_filter,kernel_size = self.filter_length3, padding='same', activation='relu',dilation_rate=1,name='conv3')(embedding)
		maxConv3 = layers.MaxPooling1D(pool_size=1,name='maxConv3')(conv3)
		x = layers.concatenate([maxConv1, maxConv2,maxConv3], axis=-1)
		# x = merge([x,],mode='cancat')
		# x = Flatten()(x)
		x = Dropout(0.5)(x)
		dense = Dense(1,activation='sigmoid')(x)
		attention1 = Flatten()(dense)
		attention1 = Activation('softmax')(attention1)
		attention1 = RepeatVector(192)(attention1)
		attention1 = Permute([2, 1], name='attention_vec1')(attention1)
		# attention1 = Flatten()(attention1)
		attention_mul1 = layers.multiply([x, attention1],name='attention_mul1')


		lstm = LSTM(self.HIDDEN_LAYER_SIZE2,dropout=0.5, recurrent_dropout=0.2,return_sequences=True)(embedding)
		drop2 = Dropout(0.5)(lstm)
		dense = Dense(1,activation='sigmoid')(drop2)
		attention2 = Flatten()(dense)
		attention2 = Activation('softmax')(attention2)
		attention2 = RepeatVector(self.HIDDEN_LAYER_SIZE2)(attention2)
		attention2 = Permute([2, 1], name='attention_vec2')(attention2)
		attention_mul2 = layers.multiply([lstm, attention2],name='attention_mul2')


		attention_mul3 = layers.add([attention_mul1, attention_mul2], name='attention_mul3')
		out_attention_mul = Flatten()(attention_mul3)
		output = Dense(1, activation='sigmoid')(out_attention_mul)
		model = Model(inputs=main_input, outputs=output)



		model.summary()
		return model