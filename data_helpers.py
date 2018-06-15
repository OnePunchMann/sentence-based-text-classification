import numpy as np
import jieba
import re
from gensim.models.word2vec import Word2Vec
import gensim

# word_len表示取最多词长度的句子为每个句子的固定长度
# 不满word_len的需要填充
def batch_maker(batch_size,word_len,data_x,data_y):
	# 载入词向量
	model = gensim.models.Word2Vec.load('./tmp/model')
	data_x_new = []
	for data in data_x:
		temp = []
		for x in range(0,len(data)):
			temp.extend(model[data[x]])
		if len(data) < word_len:
			for y in range(len(data),word_len):
				temp.extend(np.zeros(100))
		data_x_new.append(temp)
	batch_index = np.random.randint(0,len(data_x_new),batch_size)
	batch_data_x = []
	batch_data_y = []
	for num in batch_index:
		batch_data_x.append(data_x_new[num])
		batch_data_y.append(data_y[num])
	
	# for x in range(0,batch_size):
	#	print(batch_data_x[x])
	#	print(batch_data_y[x])

	return batch_data_x,batch_data_y

# 数据预处理，除去标点，去除停用词
def data_process():
	data_x = []
	data_y = []
	for num in range(10,1000):
		path = 'C:/Users/Administrator/Desktop/note/sentence-text-classification/data/C000008/'+str(num)+'.txt'
		with open(path,'r',encoding = 'ANSI')as f:
			for line in f:
				line = line.strip()
				line = re.sub(r"，|。|（|）|！|？|》|“|”|《|-|·|」|「|【|】|\?|\.|、|!|/|~|\(|\)|\||（|）|~|／|；|－|．|％|〈|〉|：|★|,|∶|\*|☆|nbsp|&|%|～|—|■|\]|\[|;|＋|●|×|…|□|◆|‰|#|◎|→|℃|\"|＊|≥|=|\s|:|‘|’|\+|○|√|※|╱|@|\\|＝|﹒|︰|△|〔|〕|>|÷|'|―|＜|＞|㎡|§|≤|`|━|［|］|{|}|＃|＆|°|﹑|≈|─|″|∈|┐|〝|〞|▲|┫","",line)
				line = list(jieba.cut(line))
				stopwords = [line.strip() for line in open('./stopwords.txt','r')]
				temp = []
				for words in line:
					if words not in stopwords:
						temp.append(words)
				line = temp
				if line and len(line) <= 50:
					data_x.append(line)
				# data_1.append([1,0])
	data_y = [[1,0] for _ in data_x]
	for num in range(10,1000):
		path = 'C:/Users/Administrator/Desktop/note/sentence-text-classification/data/C000013/'+str(num)+'.txt'
		with open(path,'r',encoding = 'ANSI')as f:
			for line in f:
				line = line.strip()
				line = re.sub(r"，|。|（|）|！|？|》|“|”|《|-|·|」|「|【|】|\?|\.|、|!|/|~|\(|\)|\||（|）|~|／|；|－|．|％|〈|〉|：|★|,|∶|\*|☆|nbsp|&|%|～|—|■|\]|\[|;|＋|●|×|…|□|◆|‰|#|◎|→|℃|\"|＊|≥|=|\s|:|‘|’|\+|○|√|※|╱|@|\\|＝|﹒|︰|△|〔|〕|>|÷|'|―|＜|＞|㎡|§|≤|`|━|［|］|{|}|＃|＆|°|﹑|≈|─|″|∈|┐|〝|〞|▲|┫","",line)								
				line = list(jieba.cut(line))
				stopwords = [line.strip() for line in open('./stopwords.txt','r')]
				temp = []
				for words in line:
					if words not in stopwords:
						temp.append(words)
				line = temp
				if line and len(line) <= 50:
					data_x.append(line)
				# data_x.append(line)
				# data_1.append([0,1])
	data_y.extend([[0,1] for _ in data_x])
	# print(data_y)
	# print(sent)
	# print(data_x[0])
	# with open('./data_x.txt','w',encoding = 'ANSI')as f:
	#	for data in data_x:
	#		for dat in data:
	#			f.write(dat)
	#			f.write(' ')
	#		f.write('\n')
	# with open('./data_y.txt','w',encoding = 'ANSI')as f:
	#	for data in data_y:
	#		for dat in data:
	#			f.write(str(dat))
	#			f.write(' ')
	#		f.write('\n')
	return data_x,data_y
def data_provider():
	with open('./data_x.txt','r',encoding = 'ANSI')as f:
		data_x = []
		for line in f:
			a = line.split()
			data_x.append(a)
	with open('./data_y.txt','r',encoding = 'ANSI')as f:
		data_y = []
		for line in f:
			temp = []
			line = re.sub(r'\s','',line)
			temp.append(int(line[0]))
			temp.append(int(line[1]))
			data_y.append(temp)
	return data_x,data_y

# 训练词向量
def w2v(data):
	model=Word2Vec(data,size = 100,window = 5,min_count = 1,iter = 10)
	model.save('./tmp/model')

def test_w2v():
	model = gensim.models.Word2Vec.load('./tmp/model')
	for word in model.wv.vocab:
		print(len(model[word]))
# max_len设定为50
def find_max_len(data):
	max = len(data[0])
	for data_x in data:
		if len(data_x) > max:
			max = len(data_x)
	return max

if __name__ == '__main__':
	# test_w2v()
	# data_x,data_y = data_process()
	# w2v(data_x)
	# batch_maker(50,50,data_x,data_y)
	data_x,data_y = data_provider()
	w2v(data_x)
	# data_process()