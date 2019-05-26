# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:25 PM

from __future__ import print_function

import nltk.stem
import numpy as np

st = nltk.stem.SnowballStemmer('english')

res_list = []
# corpus_pro为经词干化、小写化等预处理操作之后的WSDL文档
for line in open('corpus_pro.txt', 'r', encoding='utf-8'):
    temp = line.split()
    tf_dict = {}
    for item in temp:
        if item not in tf_dict:
            tf_dict[item] = 1.0/len(temp)
        else:
            tf_dict[item] = tf_dict[item] + (1.0/len(temp))
    res_list.append(tf_dict)

tag_file = r'tags_pro.txt'

mashup_tag = [[] for t in range(6206)]
index = 0

#tags.txt为标签文件
for line in open('tags.txt', 'r', encoding='utf-8'):
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index].append(st.stem(item))
    index = index + 1

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

for i in range(5170, 6206):
    tag_m = mashup_tag[i]
    tag_dict = res_list[i]
    tag_r = order_dict(tag_dict, 6)
    intersection = list((set(tag_m).union(set(tag_r))) ^ (set(tag_m) ^ set(tag_r)))
    recall = len(intersection)/len(tag_m)
    precision = len(intersection)/len(tag_r)
    R.append(recall)
    P.append(precision)

R_all = np.mean(R, axis=0)
P_all = np.mean(P, axis=0)
print('Precision：',P_all)
print('Recall：',R_all)

F = (2*R_all*P_all)/(P_all+R_all)
print('F-measure:',F)


