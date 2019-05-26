# -*- coding: UTF-8 -*-
# author: liuyang
# date: 2019/5/26
# time: 1:15 PM

from __future__ import print_function
import numpy as np
import csv
from gensim.models.keyedvectors import KeyedVectors
import numpy
from sklearn.cluster import KMeans

class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        # w2v_model对于数据集1表示以经过预处理的WSDL文档与memberAPI作为语料库训练出来的词向量模型
        # w2v_model对于数据集2表示以经过预处理的WSDL文档作为语料库训练出来的词向量模型
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass
        vector = np.mean(word_vecs, axis=0)
        return vector


    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        for doc in target_docs:
            target_vec = self.vectorize(doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append(sim_score)
            results.sort(reverse=True)

        return results


w2v_model_path = 'vectors.bin'
stopwords_path = "stopwords_en.txt"

model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

source_doc = []
docNum = 0
for line in open("merge_corpus.txt", 'r').readlines():
    source_doc.append(line.strip().lower())
    docNum=docNum+1

list = []
for i in range(0, docNum):
    print(i)
    sim_scores = ds.calculate_similarity(source_doc[i], source_doc)
    # print('\n\n')
    while(len(sim_scores)<docNum):
        sim_scores.append(0)
    list.append(sim_scores)

csvFile2 = open('w2vK_simCBOW_2.csv','w')
writer = csv.writer(csvFile2)

for i in range(0, len(list)):
    writer.writerow(list[i])


csvFile2.close()

w2v_sim_matrix = numpy.loadtxt(open("w2vK_simCBOW_2.csv","rb"), delimiter = ",", skiprows = 0)

def runKMeans(num_clusters):
     kmeans_result = KMeans(n_clusters=num_clusters, init='k-means++').fit_predict(w2v_sim_matrix)
     return kmeans_result

kmeans_result = runKMeans(20)

fk = open('w2v_km++.txt', 'w')
for item in kmeans_result:
    fk.write(str(item) + "\n")
fk.close()

# 对应的半监督聚类实验参照第二章的实验进行，或可直接使用slda的代码来进行。