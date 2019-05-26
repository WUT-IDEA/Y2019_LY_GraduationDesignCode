# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 1:14 PM

from __future__ import print_function
from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise        import cosine_similarity
from nltk.stem.lancaster             import LancasterStemmer
import scipy
import numpy as np
# from pre_process import *
from sklearn.cluster import KMeans

# stemmer
st = LancasterStemmer() #英文单词词干化

# Helper function for applying stemming(应用成熟的辅助函数）
def stemList(lin):
    lout = list()
    for item in lin:
         lout.append(st.stem(is_ascii(item)))
    return lout

# avid unicode characters（热心的unicode字符）
def is_ascii(s):
    if all(ord(c) < 128 for c in s):
        return s
    else:
        return "a"

# stemming: Bring words to their root format（词干化：将单词变成其最简形式）
def applyStemming(arg):
   stemmed = list()
   for item in arg:
       stemmed.append(" ".join(stemList(item.split())))
   return stemmed

# whole matrix print mode（整个矩阵的打印模式）
np.set_printoptions(threshold=np.inf)


# reads the data from txt file and adds each document to a list（读取TXT文件中的数据并把每个文档添加到一个列表中）
# merge_corpus.txt对于数据集1表示每条WSDL文档与对应的memberAPI的拼接，并经过相应的数据预处理。
# merge_corpus.txt对于数据集2仅表示经过数据预处理之后的WSDL文档。
with open('merge_corpus.txt') as file:
    train_set = list()
    for line in file:
        train_set.append(line.lower())

train_set_stemmed = applyStemming(train_set)

# 计算每篇文档的TF-IDF向量
vectorizer = CountVectorizer(stop_words='english')
document_term_matrix = vectorizer.fit_transform(train_set_stemmed)


tfidf = TfidfTransformer()
tfidf.fit(document_term_matrix)

# "IDF: ", tf-idf.idf_ << inverse document frequency 训练IDF模型
tf_idf_matrix = tfidf.transform(document_term_matrix)

co_sim_matrix = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

def runKMeans(num_clusters):
    kmeans_result = KMeans(n_clusters=num_clusters, init='k-means++').fit_predict(co_sim_matrix)
    return kmeans_result

kmeans_result = runKMeans(20)


fk = open('tfidf_km++.txt', 'w')
for item in kmeans_result:
    print(item)
    fk.write(str(item) + "\n")

fk.close()


# 对应的半监督聚类实验参照第二章的实验进行，或可直接使用slda的代码来进行。




