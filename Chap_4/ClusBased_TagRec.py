# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 1:26 PM

from __future__ import print_function

import nltk.stem
import numpy as np

st = nltk.stem.SnowballStemmer('english')

cluster_dict = {}
index = 0

# tf_slda.txt表示聚类的结果，可替换成通过其它聚类方法得出的聚类结果
for line in open('tf_slda.txt', 'r'):
    temp = line.strip('\n')
    cluster_dict[index] = int(temp)
    index = index + 1

# tags.txt表示所有标签
tag_path = r'tags.txt'
tag_file = open(tag_path, "r", encoding='utf-8', errors="ignores")
mashup_tag = [[] for t in range(6206)]
index2 = 0

for line in tag_file.readlines():
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index2].append(st.stem(item))
    index2 = index2 + 1

# print(mashup_tag)

rec_all = [[] for item in range(1036)]

for i in range(5170, 6206):
    for key in cluster_dict:
        if key < 5170 and cluster_dict[key] == cluster_dict[i]:
            rec_all[i-5170].extend(mashup_tag[key])

# 计算词频
rec_list = []
for temp in rec_all:
    tf_dict = {}
    for item in temp:
        if item not in tf_dict:
            tf_dict[item] = 1.0 / len(temp)
        else:
            tf_dict[item] = tf_dict[item] + (1.0 / len(temp))
    rec_list.append(tf_dict)

# print(rec_list)

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
    return res2[:n]

P = []
R = []

for i in range(1036):
    tag_m = mashup_tag[i+5170]
    tag_dict = rec_list[i]
    tag_r = order_dict(tag_dict, 12)
    intersection = list((set(tag_m).union(set(tag_r))) ^ (set(tag_m) ^ set(tag_r)))
    recall = len(intersection) / len(tag_m)
    precision = len(intersection) / len(tag_r)
    R.append(recall)
    P.append(precision)

R_all = np.mean(R, axis=0)
P_all = np.mean(P, axis=0)
print('Precision：',P_all)
print('Recall：',R_all)

F = (2*R_all*P_all)/(P_all+R_all)
print('F-measure:',F)