#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import pickle


data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"


x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb'))

def readFile():
   x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 
   return x_train,y_train,x_valid,y_valid