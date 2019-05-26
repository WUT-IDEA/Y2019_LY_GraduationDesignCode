# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 1:23 PM

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
index0 = 0
temp = []

for line in tag_file.readlines():
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index0].append(st.stem(item))
    index0 = index0 + 1
    temp.extend(temp1)

temp2 = list(set(temp))
for item in temp2:
    AlltagDict.append(st.stem(item))

# 获取learning2rank下推荐结果，结果存入列表中
ltr_list = []

score = []
# pred.txt表示有监督实验得出的打分结果
for line in open('pred.txt', 'r', encoding='utf-8'):
    temp = line.replace('[', '').replace(']', '').strip()
    score.append(float(temp))

index1 = 0

for i in range(5170, 6206):
    # 正样本
    tag_dict = {}
    tempTags = list(AlltagDict)
    mashup_tag[i] = list(set(mashup_tag[i]))
    for t in mashup_tag[i]:
        tag_dict[t]=score[index1]
        index1 = index1 + 1

    for item in tempTags[:]:
        if item in mashup_tag[i]:
            tempTags.remove(item)

    for t in tempTags:
        tag_dict[t] = score[index1]
        index1 = index1 + 1
    ltr_list.append(tag_dict)


# 获取聚类的下的推荐结果，结果存入列表中
cluster_dict = {}
index2 = 0

# tfidf_km++.txt表示聚类结果，也可替换成由其它聚类方法得出的聚类结果
for line in open('tfidf_km++.txt', 'r'):
    temp = line.strip('\n')
    cluster_dict[index2] = int(temp)
    index2 = index2 + 1

tag_path = r'tags.txt'
tag_file = open(tag_path, "r", encoding='utf-8', errors="ignores")
mashup_tag = [[] for t in range(6206)]
index3 = 0

for line in tag_file.readlines():
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[index3].append(st.stem(item))
    index3 = index3 + 1

# print(mashup_tag)

rec_all = [[] for item in range(1036)]

for i in range(5170, 6206):
    for key in cluster_dict:
        if key < 5170 and cluster_dict[key] == cluster_dict[i]:
            rec_all[i-5170].extend(mashup_tag[key])

# 计算词频
clu_list = []
for temp in rec_all:
    tf_dict = {}
    for item in temp:
        if item not in tf_dict:
            tf_dict[item] = 1.0 / len(temp)
        else:
            tf_dict[item] = tf_dict[item] + (1.0 / len(temp))
    clu_list.append(tf_dict)

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

def borda_count(rec1, rec2, n):
    l1 = len(rec1)
    l2 = len(rec2)
    sco_dict1 = {}
    sco_dict2 = {}
    # sco_dict = {}
    for i in rec1:
        sco_dict1[i] = l1
        l1 = l1 - 1
    for i in rec2:
        sco_dict2[i] = l2
        l2 = l2 - 1
    for key in sco_dict2:
        if key in sco_dict1:
            sco_dict1[key] = sco_dict1[key] + sco_dict2[key]
        else:
            sco_dict1[key] = sco_dict2[key]

    return order_dict(sco_dict1, n)


P = []
R = []

for i in range(5170, 6206):
    tag_m = mashup_tag[i]
    tag_dict1 = ltr_list[i - 5170]
    tag_dict2 = clu_list[i-5170]
    # 下句中的前两个参数为K，在融合实验中是一个可调的参数，最后一个参数表示融合方法中推荐的标签数目
    tag_r = borda_count(order_dict(tag_dict1, 100), order_dict(tag_dict2, 100), 6)
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