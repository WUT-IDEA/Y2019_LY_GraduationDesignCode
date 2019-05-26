# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:31 PM

from __future__ import print_function

import lda.datasets
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import numpy
import nltk.stem

st = nltk.stem.SnowballStemmer('english')

model_corpus = []
# model_data_lda.txt为经预处理之后的WSDL文档和标签拼接之后形成的语料
for line in open('model_data_lda.txt', 'r', encoding='utf-8'):
    model_corpus.append(line.strip())

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(model_corpus)
analyze = vectorizer.build_analyzer()
weight = x.toarray()

model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(np.asarray(weight))  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
# 文档-主题（Document-Topic）分布
doc_topic = model.doc_topic_
a = doc_topic
#
# numpy.savetxt('/Users/liuyang/PycharmProjects/tensorflow-LTR/Comparative Experiment/lda_matrix.csv', a, delimiter=',')

# print(len(a))
# print(a)

tag_dict = {}
index1 = 6206
all_tag = []
# tags_pro3.txt为经预处理之后的标签词典
for line in open('tags_pro3.txt', 'r', encoding='utf-8'):
    tag_dict[line.replace('\n', '')] = index1
    all_tag.append(line.replace('\n', ''))
    index1 = index1 + 1

# tags.txt为原始标签集合
tag_path = r'tags.txt'
tag_file = open(tag_path, "r", encoding='utf-8', errors="ignores")
mashup_tag = [[] for t in range(6206)]
index2 = 0

for line in tag_file.readlines():
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index2].append(st.stem(item))
    index2 = index2 + 1


def cosine(_vec1, _vec2):
  return float(numpy.sum(_vec1*_vec2))/(numpy.linalg.norm(_vec1)*numpy.linalg.norm(_vec2))

res_list = []

for i in range(5170, 6206):
    tag_sim = {}
    for t in all_tag:
        tag_sim[t] = cosine(a[i], a[tag_dict[t]])
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
    tag_r = order_dict(tag_dict, 5)
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




