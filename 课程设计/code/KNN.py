#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ReadData

      # 输出分类信息
# 峰值0.484
if __name__ == "__main__":
    x_train,y_train,x_valid,y_valid = ReadData.readFile()
    #文本特征向量化
    vec = CountVectorizer()
    X_train1 = vec.fit_transform(x_train)
    X_test = vec.transform(x_valid)
    resultDict = dict()
    for i in range(100):
        model = KNeighborsClassifier(n_neighbors=(i+1))
        model.fit(X_train1, y_train)
        predicted = model.predict(X_test)
        resultDict[i+1] = sum(predicted == y_valid) / len(y_valid)
        print(sum(predicted == y_valid) / len(y_valid))
    
    print(sorted(resultDict.items()))