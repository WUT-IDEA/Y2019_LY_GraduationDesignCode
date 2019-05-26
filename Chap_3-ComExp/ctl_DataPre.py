# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 12:35 PM

from __future__ import print_function
import nltk.stem
import numpy

extr_list = []
new_tag = []
for line in open('corpus_pro.txt', 'r', encoding='utf-8'):
    temp = line.split()
    tf_dict = {}
    for item in temp:
        if item not in tf_dict:
            tf_dict[item] = 1.0/len(temp)
        else:
            tf_dict[item] = tf_dict[item] + (1.0/len(temp))
    extr_list.append(tf_dict)

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

for i in range(5170, 6206):
    test_tag = order_dict(extr_list[i], 6)
    new_tag.extend(test_tag)
new_tag = list(set(new_tag))
index = 0
tag_dict = {}
for item in new_tag:
    tag_dict[item] = index
    index = index + 1

######################
# 训练集构造
st = nltk.stem.SnowballStemmer('english')

# 给标签标ID
for line in open("tags_pro3.txt"):
    temp = line.strip().replace("\n", "")
    if temp not in tag_dict:
        tag_dict[temp] = index
        index = index + 1

mashup_tag = [[] for t in range(6206)]
masht_index = 0
for line in open("tags.txt"):
    temp1 = line.replace('\n', '').lower().split('###')
    for item in temp1:
        mashup_tag[masht_index].append(st.stem(item))
    masht_index = masht_index + 1

# print(mashup_tag[0])
train_tag_vec = [[] for i in range(5170)]
for i in range(5170):
    temp1 = ''
    DocNumLabels = len(list(set(mashup_tag[i])))
    # train_tag_vec[i].append(str(DocNumLabels))
    temp1 = str(DocNumLabels)
    for t in list(set(mashup_tag[i])):
        temp1 = temp1 + ' ' + str(tag_dict[t])
    train_tag_vec[i].append(temp1)
    # print(temp1)

# print(train_tag_vec)

_corpus_dict = {}
train_corpus = []
cor_index = 0
for line in open("corpus_pro.txt"):
    temp = line.replace('\n', '').split()
    train_corpus.extend(temp)

# print(list(set(train_corpus)))
# print(len(list(set(train_corpus))))
for item in list(set(train_corpus)):
    _corpus_dict[item] = cor_index
    cor_index = cor_index + 1

mashup_corpus = [[] for c in range(6206)]
mashc_index = 0
for line in open("corpus_pro.txt"):
    mashup_corpus[mashc_index] = line.strip().split()
    mashc_index = mashc_index + 1


def count_num(corpus_list):
    count_dict = {}
    for item in corpus_list:
        if item in count_dict:
            count_dict[item] = count_dict[item] + 1
        else:
            count_dict[item] = 1
    return count_dict


train_corpus_vec = []
for i in range(5170):
    DocNumWords = len(list(set(mashup_corpus[i])))
    corpus_temp = []
    temp1 = str(DocNumWords)
    # corpus_temp.append(str(DocNumWords))
    for c in list(set(mashup_corpus[i])):
        temp = str(_corpus_dict[c]) + ':' + str(count_num(mashup_corpus[i])[c])
        temp1 = temp1 + ' ' + temp
    corpus_temp.append(temp1)
    train_corpus_vec.append(corpus_temp)

train_vec = []

for i in range(5170):
    temp = ''.join(train_tag_vec[i]) + ' @ ' + ''.join(train_corpus_vec[i])
    # print(temp)
    train_vec.append(temp)

# print(train_vec)
# numpy.savetxt('/Users/liuyang/PycharmProjects/tensorflow-LTR/Comparative Experiment/ctl_train_corpus.txt', train_vec, delimiter = ' ', fmt='%s')




test_tag_vec = []
for i in range(5170, 6206):
    test_tag = order_dict(extr_list[i], 6)
    DocNumLabels = len(test_tag)
    temp1 = str(DocNumLabels)
    for t in test_tag:
        temp1 = temp1 + ' ' + str(tag_dict[t])
    test_tag_vec.append(temp1)
# print(test_tag_vec)

test_corpus_vec = []
for i in range(5170, 6206):
    DocNumWords = len(list(set(mashup_corpus[i])))
    corpus_temp = []
    temp1 = str(DocNumWords)
    # corpus_temp.append(str(DocNumWords))
    for c in list(set(mashup_corpus[i])):
        temp = str(_corpus_dict[c]) + ':' + str(count_num(mashup_corpus[i])[c])
        temp1 = temp1 + ' ' + temp
    corpus_temp.append(temp1)
    test_corpus_vec.append(corpus_temp)
# print(test_corpus_vec)

test_vec = []
for i in range(5170, 6206):
    temp = ''.join(test_tag_vec[i-5170]) + ' @ ' + ''.join(test_corpus_vec[i-5170])
    # print(temp)
    test_vec.append(temp)
for line in open("/Users/liuyang/PycharmProjects/tensorflow-LTR/Comparative Experiment/tags_pro3.txt"):
    temp1 = line.strip().replace("\n", "")
    temp = str(1) + ' ' + str(tag_dict[temp1]) + ' @ ' + str(1) + ' ' + str(tag_dict[temp1]) + ':' + str(1)
    # temp = ''.join(tag_dict[temp1]) + ' @ ' + ''.join(tag_dict[temp1]) + ':' + ''.join(str(1))
    test_vec.append(temp)
    # print(temp)

numpy.savetxt('/Users/liuyang/PycharmProjects/tensorflow-LTR/Comparative Experiment/ctl_test_corpus.txt', test_vec, delimiter = ' ', fmt='%s')


# 该部分数据处理结束之后，结合CTL文件夹中的程序进行操作，最终获得文本-主题分布作为本论文中ctl对比实验的特征提取方案
