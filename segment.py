import jieba

file = open('sum_en.txt',encoding='utf-8',mode='r')
file2 = open('sum_seg.txt',encoding='utf-8',mode='w')

def cut_sentence(sentence):
	words = jieba.cut(sentence, cut_all=False)
	return list(words)

for line in file.readlines():
	words = cut_sentence(line.strip())
	for w in words:
		file2.write(w)
		file2.write(' ')
	file2.write('\n')
