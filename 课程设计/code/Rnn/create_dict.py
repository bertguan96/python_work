#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 创建数据字典
import os
import pickle
import numpy as np
from create_data import participle_to_label
from create_data import tf_method
import json

data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\"
# 读取参数
x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 


""" 生成数据字典 """
def init_dict(data_list):
    # 调用字典转换方法
    word_dict,participle_arr = participle_to_label(data_list) 
    f = open(root + "dict_txt_all.txt","w")
    f.write(str(word_dict))
    print(len(data_list))

"""生成训练文本集合"""
def create_list(docList, fileName, dicts):
    word_dict,participle_arr = participle_to_label(docList)
    tf_method(dicts,participle_arr,fileName,data_path=root)

def create_valid_text():
    f = open(root + "valid.txt","w")
    for text in x_valid:
        f.write(str(text) + "\n")
    print("创建验证集合成")

"""创建测试集合代码"""
def create_test_test(x_text):
    f = open(root + "test.txt","w")
    for text in x_text:
        f.write(str(text) + "\n")
    print("创建测试集合成功！")

# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])

    return len(line.keys())

if __name__ == "__main__":
    data_list = list(x_train)
    # 将验证集合代码加入list里面
    for validStr in x_valid:
        data_list.append(validStr)
    # 将测试集合的代码加入List里面
    # 生成数据字典
    init_dict()
    f = open(root + "dict_txt_all.txt","r")
    dicts = f.read()
    dictinfo = eval(dicts)

    train_dict_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\train_list.txt"
    valid_dict_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\valid_list.txt"

    # 生成训练集合
    # create_list(list(x_train),"train_list.txt",dictinfo)


    # files1 = open(root+"train_list_end.txt", "w",encoding="utf-8")
   

    # i = 0
    # for files in open(train_dict_path):
    #     label = str(y_train[i])
    #     strClass = ",".join(files.strip("\n").split())
    #     str2 = strClass + "\t" + str(label) + "\n"
    #     files1.write(str2)  
    #     print("step i is :%s" % i)
    #     i+=1
    files2 = open(root+"valid_list_end.txt", "w",encoding="utf-8")
    # 生成测试集合
    create_list(list(x_valid),"valid_list.txt",dictinfo) 

    j = 0
    for files in open(valid_dict_path):
        label = str(y_valid[j])
        strClass = ",".join(files.strip("\n").split())
        str2 = strClass + "\t" + str(label) + "\n"
        files2.write(str2)  
        print("step j is :%s" % j)
        j+=1

    create_valid_text()
    # 需要验证代码时候启用
    # create_test_test(text_list)