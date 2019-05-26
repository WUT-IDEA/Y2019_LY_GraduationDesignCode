# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:54 PM

from __future__ import print_function

import gensim
import numpy as np
import random
import numpy
import nltk.stem

model = gensim.models.Word2Vec.load('total.model')
st = nltk.stem.SnowballStemmer('english')

# 文本向量：词向量累加
def fea_vec(s, model):
    vecs = [a for a in filter(None, s.split())]
    # print(vecs)
    fea_vecs = []
    for item in vecs:
        vec = model[item]
        fea_vecs.append(vec)
    vector = np.sum(fea_vecs, axis=0)
    return vector

# desc部分特征构造
fcp = open('corpus_pro.txt', 'r')

corpus_matrix = []

for line in fcp.readlines():
    line = line.strip('\n')
    corpus_matrix.append(fea_vec(line, model))
#
# print(corpus_matrix)


# API部分特征构造，两个数据集中，一个有memberAPI这个特征一个没有，有的就加上这部分，没有就去掉
fAPI = open('API_pro.txt', 'r')

API_matrix = []

for line in fAPI.readlines():
    line = line.strip('\n')
    API_matrix.append(fea_vec(line, model))
#
# print(API_matrix)


# 特征拼接
# 构造所有标签的词典，以及每个mashup对应的真实标签集
tag_path = r'tags.txt'
tag_file = open(tag_path, "r", encoding='utf-8', errors="ignores")

AlltagDict = []
mashup_tag = [[] for t in range(6206)]
index = 0
temp = []

for line in tag_file.readlines():
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index].append(st.stem(item))
    index = index + 1
    temp.extend(temp1)
# print(mashup_tag)
# print(len(mashup_tag))

temp2 = list(set(temp))
for item in temp2:
    AlltagDict.append(st.stem(item))


# print(AlltagDict)
# print(len(AlltagDict))

ftags = open('tags_pro3.txt', 'w')

for item in AlltagDict:
    ftags.write(item + '\n')

ftags.close()

for i in range(6206):
    mashup_tag[i] =  list(set(mashup_tag[i]))


def cosine(_vec1, _vec2):
  return float(numpy.sum(_vec1*_vec2))/(numpy.linalg.norm(_vec1)*numpy.linalg.norm(_vec2))

tag_freq = dict()

all_tag = []

for i in range(6206):
    all_tag.extend(mashup_tag[i])

total_tag = len(all_tag)
for item in all_tag:
    if item not in tag_freq:
        tag_freq[item] = 1.0/total_tag
    else:
        tag_freq[item] = tag_freq[item] + (1.0/total_tag)

index = 1
for i in range(6206):
    for item in mashup_tag[i]:
        # print(str(index)+':'+str(tag_freq[item]))
        index = index + 1

# 构造训练集、验证集和测试集，比例为4：1：1

train_data = []
vali_data = []
test_data = []
#
for i in range(0, 4136):
    tempTags = list(AlltagDict)
    mashup_tag[i] = list(set(mashup_tag[i]))

    # 正样本
    for t in mashup_tag[i]:
        temp = list(corpus_matrix[i]) + list(API_matrix[i]) + list(fea_vec(t, model)) + list([tag_freq[t]]) + list([cosine(corpus_matrix[i], fea_vec(t, model))])
        # print(temp)
        train_temp = list(str(1))+ list(map(lambda a, b: str(a) + ':' + str(b), ['qid'], [i])) + list(map(lambda x, y: str(x) + ':' + str(y), [x for x in range(1, len(temp)+1)], temp))
        train_data.append(train_temp)


    for item in tempTags[:]:
        if item in mashup_tag[i]:
            tempTags.remove(item)

    slice = random.sample(tempTags, len(mashup_tag[i]))

    # 负样本
    for t in slice:
        temp = list(corpus_matrix[i]) + list(API_matrix[i]) + list(fea_vec(t, model)) + list([tag_freq[t]]) + list([cosine(corpus_matrix[i], fea_vec(t, model))])
        train_temp = list(str(0)) + list(map(lambda a, b: str(a) + ':' + str(b), ['qid'], [i])) + list(map(lambda x, y: str(x) + ':' + str(y), [x for x in range(1, len(temp)+1)], temp))
        train_data.append(train_temp)

numpy.savetxt('train.txt', train_data, delimiter = ' ', fmt='%s')



for i in range(4136, 5170):
    # 正样本
    mashup_tag[i] = list(set(mashup_tag[i]))

    for t in mashup_tag[i]:
        temp = list(corpus_matrix[i]) + list(API_matrix[i]) + list(fea_vec(t, model)) + list([tag_freq[t]]) + list([cosine(corpus_matrix[i], fea_vec(t, model))])
        vali_temp = list(str(1)) + list(map(lambda a, b: str(a) + ':' + str(b), ['qid'], [i])) + list(map(lambda x, y: str(x) + ':' + str(y), [x for x in range(1, len(temp)+1)], temp))
        vali_data.append(vali_temp)

    negsample = [t for t in AlltagDict if not t in mashup_tag[i]]

    for t in negsample:
        temp = list(corpus_matrix[i]) + list(API_matrix[i]) + list(fea_vec(t, model)) + list([tag_freq[t]]) + list([cosine(corpus_matrix[i], fea_vec(t, model))])
        vali_temp = list(str(0)) + list(map(lambda a, b: str(a) + ':' + str(b), ['qid'], [i])) + list(map(lambda x, y: str(x) + ':' + str(y), [x for x in range(1, len(temp)+1)], temp))
        vali_data.append(vali_temp)
#
# print(vali_data)
numpy.savetxt('vali.txt', vali_data, delimiter = ' ', fmt='%s')



for i in range(5170, 6206):
    # 正样本
    tag_dict = {}
    tempTags = list(AlltagDict)
    mashup_tag[i] = list(set(mashup_tag[i]))
    for t in mashup_tag[i]:
        temp = list(corpus_matrix[i]) + list(API_matrix[i]) + list(fea_vec(t, model)) + list([tag_freq[t]]) + list([cosine(corpus_matrix[i], fea_vec(t, model))])
        test_temp = list(str(1)) + list(map(lambda a, b: str(a) + ':' + str(b), ['qid'], [i])) + list(map(lambda x, y: str(x) + ':' + str(y), [x for x in range(1, len(temp)+1)], temp))
        test_data.append(test_temp)


    for item in tempTags[:]:
        if item in mashup_tag[i]:
            tempTags.remove(item)

    for t in tempTags:
        temp = list(corpus_matrix[i]) + list(API_matrix[i]) + list(fea_vec(t, model)) + list([tag_freq[t]]) + list([cosine(corpus_matrix[i], fea_vec(t, model))])
        test_temp = list(str(0)) + list(map(lambda a, b: str(a) + ':' + str(b), ['qid'], [i])) + list(map(lambda x, y: str(x) + ':' + str(y), [x for x in range(1, len(temp)+1)], temp))
        test_data.append(test_temp)

numpy.savetxt('test.txt', test_data, delimiter = ' ', fmt='%s')
