#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

# ===============================样本不平衡、多分类的情况========================
import numpy as np
import handle
from sklearn import svm
import random
from sklearn import metrics
import numpy



if __name__ == '__main__':
    # 创建一个向量集合存放处理好的测试集合的文本向量
    train_words,train_tags  = handle.getTrainData()
    # 解决维度问题抛出的错误(这里需要对数据reshape一下)
    test_words,test_tags = handle.getTestData()

    X = train_words
    y = train_tags
    clt = svm.SVC(gamma="auto",kernel='rbf')
    print(clt)
    clt.fit(X, y)
    print(clt)
    re =  clf.predict(np.array(test_words[0]).reshape(1,-1))
    # evaluate(numpy.asarray(test_tags),re)
    print(re)
