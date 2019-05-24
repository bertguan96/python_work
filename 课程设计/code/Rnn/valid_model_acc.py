#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# 准确率校验方法

import pickle
from sklearn.metrics import accuracy_score

data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\Rnn\\result\\"

# file result_cnn_0524.txt 0.987

x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 

y_test = list(y_valid)

y_predict = list()

for y in open(root + 'result_cnn_0524.txt','r'):
    y = str(y).strip('\n')   # 去除末尾换行符
    y_predict.append(int(y))

print(accuracy_score(y_predict, y_test))