#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# 准确率校验方法

import pickle
from sklearn.metrics import accuracy_score

data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
    
root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\Rnn\\datasets\\"



# file result_cnn_0524.txt 0.987

def get_acc():
    x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 
    y_test = list(y_valid)
    # y_test = y_test[499:]

    y_predict = list()
    for y in open(root + 'result1.txt','r'):
        y = str(y).strip('\n')   # 去除末尾换行符
        y_predict.append(int(y))
    print(accuracy_score(y_predict, y_test))


# output = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\Rnn\\datasets1\\"

# def get_acc1():
#     y_valid = []
#     for label in open(output+"label.txt","r",encoding="utf-8"):
#         y_valid.append(int(str(label).strip("\n")))
#     y_predict = list()
#     for y in open(output + 'result1.txt','r'):
#         y = str(y).strip('\n')   # 去除末尾换行符
#         y_predict.append(int(y))
#     print(accuracy_score(y_predict, y_valid))
get_acc()