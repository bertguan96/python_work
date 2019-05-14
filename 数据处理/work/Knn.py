#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import os
import numpy as np
import handle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


      # 输出分类信息

if __name__ == "__main__":
    # 创建一个向量集合存放处理好的测试集合的文本向量
    train_data_set,classVec  = handle.getTrainData()
    # 解决维度问题抛出的错误(这里需要对数据reshape一下)
    thisDoc,label_name = handle.getTestData()
    # 交叉分类
    train_X = train_data_set
    test_X = classVec
    # 验证集合
    train_y = thisDoc 
    # KNN模型，选择3个邻居

    i = 0
    nList = [51]
    for n in nList:
        f = open("E:\\PythonProject\\大数据处理与实践\\result_knn_"+str(n)+".txt","w",encoding="utf-8")
        for doc in train_y:
            model = KNeighborsClassifier(n_neighbors=n)
            model.fit(train_X, test_X)
            predicted = model.predict(np.array(doc).reshape(1,-1))
            if label_name[i]/10 < 1:
                file_name = "00000" + str(label_name[i])
            elif label_name[i]/100 < 1:
                file_name = "0000" + str(label_name[i])
            elif label_name[i]/1000 < 1:
                file_name = "000" + str(label_name[i])
            elif label_name[i]/10000 < 1:
                file_name = "00" + str(label_name[i])
            elif label_name[i]/100000 < 1:
                file_name = "0" + str(label_name[i])
            elif label_name[i]/1000000 < 1:
                file_name = str(label_name[i])
            print(file_name)
            f.write(file_name+"\t00"+str(predicted[0])+"\n")
            i+=1