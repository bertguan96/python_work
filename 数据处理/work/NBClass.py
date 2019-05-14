#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import os
import numpy as np
import handle
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    # 创建一个向量集合存放处理好的测试集合的文本向量
    train_data_set,classVec  = handle.getTrainData()
    # 解决维度问题抛出的错误(这里需要对数据reshape一下)
    thisDoc,label_name = handle.getTestData()
    print()
    f = open("E:\\PythonProject\\大数据处理与实践\\result.txt","w",encoding="utf-8")
    # 导入sklearn包 开始进行计算
    i = 0
    for doc in thisDoc:
        clf = GaussianNB()
        y_pred = clf.fit(np.array(train_data_set), classVec)
        x = y_pred.predict(np.array(doc).reshape(1,-1))
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
        f.write(file_name+"\t"+str(x[0])+"\n")
        i+=1