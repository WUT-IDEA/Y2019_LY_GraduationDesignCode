# Mashup服务聚类与标签推荐实验

## Chap_2
本章实验包括对Mashup服务进行聚类与半监督聚类的实验。该部分内容已经在小论文的Demo里面展示过。可以直接参照[sldaCluster](https://github.com/liuyangzy/sldaCluster.git)，聚类之后的内容参考了David Blei大佬实验室的[code](https://www.cs.princeton.edu/~blei/topicmodeling.html ),可以自行尝试聚类之后的半监督过程。该代码的使用方法可参照[sLDA](https://blog.csdn.net/houmou/article/details/49532673)的使用博客。

## Chap_3-ComExp
该部分为第三章的对比实验，包括四个Mashup服务标签推荐的对比方法。方法代码都在这里，具体的数据预处理需要自己操作，但代码中对用到的数据经过了什么样的预处理操作大体都有解释。

另外，ctl实验，用到了大佬的[code](https://github.com/shuangyinli/CTL.git)

## Chap_3-Supervised
该部分为基于有监督方法的服务标签推荐实验，基于Learning2Rank方法，可以基于数据的特征来选择具体的方法，该部分代码参照了大佬们的[code](https://github.com/ChenglongChen/tensorflow-LTR.git)

基于有监督方法进行实验之后，有一个特征优化的过程，在代码中没有体现出来，不过认真阅读之后应该会知道优化特征该在哪里添加。

## Chap_4
本章实验包括基于聚类的服务标签推荐实验和融合实验。根据集成学习的原则，选取差异较大但是各自性能较优的基学习器来进行融合。该部分实验有一个调参的过程，在代码中没有体现出来，但认真阅读之后应该会知道在哪里调参。

## 总结
代码不够美观，只能作为实验思路的一种参考。


