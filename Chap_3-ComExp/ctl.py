# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:38 PM

from __future__ import print_function
from numpy import *
import numpy
import nltk.stem
import numpy as np

st = nltk.stem.SnowballStemmer('english')

A = zeros((6609, 20), dtype=float)

f = open("doc-topics-dis.txt")

lines = f.readlines()
A_row = 0
for line in lines:
    li = line.strip('\n').split()
    A[A_row:] = li[0:20]
    A_row = A_row + 1
# print(len(A))


tag_dict = {}
index1 = 6206
all_tag = []
for line in open('tags_pro3.txt', 'r', encoding='utf-8'):
    tag_dict[line.replace('\n', '')] = index1
    all_tag.append(line.replace('\n', ''))
    index1 = index1 + 1

tag_path = r'tags.txt'
tag_file = open(tag_path, "r", encoding='utf-8', errors="ignores")
mashup_tag = [[] for t in range(6206)]
index2 = 0

for line in tag_file.readlines():
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index2].append(st.stem(item))
    index2 = index2 + 1
# print(len(mashup_tag))


def cosine(_vec1, _vec2):
  return float(numpy.sum(_vec1*_vec2))/(numpy.linalg.norm(_vec1)*numpy.linalg.norm(_vec2))

# 每个文档与对应403个标签的相似度字典
res_list = []

for i in range(5170, 6206):
    tag_sim = {}
    for t in all_tag:
        tag_sim[t] = cosine(A[i], A[tag_dict[t]])
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
    # print(tag_m)
    tag_dict = res_list[i-5170]
    tag_r = order_dict(tag_dict, 6)
    ret_list = list((set(tag_m).union(set(tag_r))) ^ (set(tag_m) ^ set(tag_r)))
    recall = len(ret_list)/len(tag_m)
    precision = len(ret_list)/len(tag_r)
    R.append(recall)
    P.append(precision)

R_all = np.mean(R, axis=0)
P_all = np.mean(P, axis=0)

print('Recall：',R_all)
print('Precision：',P_all)
F = (2*R_all*P_all)/(P_all+R_all)
print('F-measure:',F)

