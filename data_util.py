import gensim
import collections  # 用来统计词频
import numpy as np
import jieba
import pickle
import codecs
import nltk
from keras import backend as K
from keras.models import load_model, Model
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

MAX_FEATURES = 100000
MAX_SENTENCE_LENGTH = 300
ALL_VECTOR = 160000


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


stopwords = []

def get_stop_words():
	fr = open('stopword.txt',encoding='utf-8',mode='r')
	for line in fr.readlines():
		line = line.strip()
		stopwords.append(line)
	return stopwords


def cut_sentence(sentence):
	words = jieba.cut(sentence, cut_all=False)
	words_clean = []
	for w in words:
		if w  not in stopwords:
			words_clean.append(w)
	# print(words_clean)
	return words_clean


def embeddings2index(word2vec_path):
	embeddings_index = {}
	model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
	word_vectors = model.wv
	for word, vocab_obj in model.wv.vocab.items():
		if int(vocab_obj.index) < ALL_VECTOR:
			embeddings_index[word] = word_vectors[word]
	print('Found %s word vectors.' % len(embeddings_index))
	# 删掉gensim模型释放内存
	del model, word_vectors
	print(len(embeddings_index))
	return embeddings_index


def get_feature_vec(data, model_path):
	model = load_model(model_path, custom_objects={"f1": f1})
	out_attention_mul_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
	feature_vec = out_attention_mul_layer_model.predict(data)
	return feature_vec

def sentences2vec(sentences, word2index_path,model_path):
	word2index = pickle.load(word2index_path)
	sen_num = len(sentences)
	X = np.empty(sen_num, dtype=list)

	count = 0
	for line in sentences:
		# words = nltk.word_tokenize(line.lower())
		words = cut_sentence(line.strip())
		seqs = []
		for word in words:
			if word in word2index:
				seqs.append(word2index[word])
			else:
				seqs.append(word2index["UNK"])
		X[count] = seqs
		count += 1
	X = get_feature_vec(X,model_path)
	return X


def data_preprocess(file_path, word2vec_path, word2index_path, embedding_size=128):
	maxlen = 0
	word_freqs = collections.Counter()
	num_recs = 0
	# 从文件中读取训练以及测试数据
	positive_examples = list(open(file_path, mode="r", encoding="utf-8").readlines())
	positive_examples = [s.strip() for s in positive_examples]
	for line in positive_examples:
		# 分割标签和内容
		label, sentence = line.strip().split("/t")
		# words = nltk.word_tokenize(sentence.strip().lower())
		words = cut_sentence(sentence.strip())
		if len(words) > maxlen:
			maxlen = len(words)
		for word in words:
			word_freqs[word] += 1
		num_recs += 1


	embeddings_index = embeddings2index(word2vec_path=word2vec_path)


	vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
	word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
	word2index["PAD"] = 0
	word2index["UNK"] = 1

	fw = codecs.open(word2index_path, mode='wb')
	pickle.dump(word2index, fw)
	index2word = {v: k for k, v in word2index.items()}
	X = np.empty(num_recs, dtype=list)
	y = np.zeros(num_recs)

	count = 0

	for line in positive_examples:
		label, sentence = line.strip().split("/t")
		# words = nltk.word_tokenize(sentence.strip().lower())
		words = cut_sentence(sentence.strip())
		seqs = []
		for word in words:
			if word in word2index:
				seqs.append(word2index[word])
			else:
				seqs.append(word2index["UNK"])
		X[count] = seqs
		y[count] = int(label)
		count += 1

	# 补齐所有数据到相同的长度，不足长的在前面补0，超长的在开始处截断
	X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
	# 数据划分为训练集和测试集
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)

	# 对比词向量字典中包含词的个数与文本数据所有词的个数，取小
	num_words = min(vocab_size, len(word_freqs))
	embedding_matrix = np.zeros((num_words+2, embedding_size))
	print("embedding_matrix.shape:", embedding_matrix.shape)
	for word,i in word2index.items():
		if i >= ALL_VECTOR:
			continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		else:
			embedding_matrix[i] = np.zeros([128])
	return num_words+2, embedding_matrix, Xtrain, ytrain, Xtest, ytest


if __name__ == '__main__':
	sen = "发现这家公园评价这家公园童年回忆小时候住记得小时候人气旺老爷爷老奶奶喜欢锻炼身体记得有颗树特别小朋友做秋千吃早饭怀念味道这家公园总体不错"
	cut_sentence(sen)
