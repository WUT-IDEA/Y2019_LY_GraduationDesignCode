# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:49 PM

from __future__ import print_function

import numpy as np
import nltk.stem

st = nltk.stem.SnowballStemmer('english')

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

temp2 = list(set(temp))
for item in temp2:
    AlltagDict.append(st.stem(item))

res_list = []

score = []
for line in open('pred.txt', 'r', encoding='utf-8'):
    temp = line.replace('[', '').replace(']', '').strip()
    score.append(float(temp))

index = 0

for i in range(5170, 6206):
    # 正样本
    tag_dict = {}
    tempTags = list(AlltagDict)
    mashup_tag[i] = list(set(mashup_tag[i]))
    for t in mashup_tag[i]:
        tag_dict[t]=score[index]
        index = index + 1

    for item in tempTags[:]:
        if item in mashup_tag[i]:
            tempTags.remove(item)

    for t in tempTags:
        tag_dict[t] = score[index]
        index = index + 1
    res_list.append(tag_dict)

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
