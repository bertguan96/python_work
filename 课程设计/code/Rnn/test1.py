#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT


import wordninja as wdja
import pickle
import wordcheck

data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"


root = "课程设计\\code\\Rnn\\datasets\\"

x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb'))


x_train_file = open(root+"trian.txt","r")
x_train_file1 = open(root+"train_dict.txt","w")
i = 0
res = []
for data in x_train_file:
    res = str(data).strip("\n") + "\t" + str(y_train[i])
    x_train_file1.write(res + "\n")
    i+=1
    
