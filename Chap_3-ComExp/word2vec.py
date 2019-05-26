# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:29 PM

from __future__ import print_function

import gensim
import numpy as np
import numpy
import nltk.stem

# w2v_corpusAndtags.model为经预处理之后的WSDL文档和tags的合并语料训练成的词向量模型
model = gensim.models.Word2Vec.load('w2v_corpusAndtags.model')
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
# corpus_pro.txt为经预处理的WSDL文档
fcp = open('corpus_pro.txt', 'r')

corpus_matrix = []

for line in fcp.readlines():
    line = line.strip('\n')
    corpus_matrix.append(fea_vec(line, model))


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


def cosine(_vec1, _vec2):
  return float(numpy.sum(_vec1*_vec2))/(numpy.linalg.norm(_vec1)*numpy.linalg.norm(_vec2))



res_list = []

for i in range(5170, 6206):
    tag_sim = {}

    for t in AlltagDict:
        tag_sim[t] = cosine(corpus_matrix[i], fea_vec(t, model))

    res_list.append(tag_sim)


def order_dict(dicts, n):
    res1 = []
    res2 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1]==i:
                res1.append(j)
    for r in res1:
        res2.append(r[0])
    return res2


P = []
R = []

for i in range(5170, 6206):
    tag_m = mashup_tag[i]
    tag_dict = res_list[i - 5170]
    tag_r = order_dict(tag_dict, 3)
    ret_list = list((set(tag_m).union(set(tag_r))) ^ (set(tag_m) ^ set(tag_r)))
    recall = len(ret_list)/len(tag_m)
    precision = len(ret_list)/len(tag_r)
    R.append(recall)
    P.append(precision)

R_all = np.mean(R, axis=0)
P_all = np.mean(P, axis=0)

print('平均召回率：',R_all)
print('平均精确率：',P_all)
F = (2*R_all*P_all)/(P_all+R_all)
print('F-measure:',F)

