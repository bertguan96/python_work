#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# 准确率校验方法

import pickle
from sklearn.metrics import accuracy_score

data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\Rnn\\result\\"

# file result_cnn acc 0.803
# file result_bli_cnn acc 0.804

x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 

y_test = list(y_valid)

y_predict = list()

for y in open(root + 'result_cnn.txt','r'):
    y = str(y).strip('\n')   # 去除末尾换行符
    y_predict.append(int(y))

print(accuracy_score(y_predict, y_test))