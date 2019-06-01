#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 创建数据字典
import os
import pickle
import numpy as np
from create_data_v1 import participle_to_label
from create_data_v1 import tf_method
import read_config
import json
import time
import random

# 设置读取类型
TRIAN_TYPE = "train"
TEST_TYPE = "test"
# 训练集路径
train_path = read_config.ReadConfig(TRIAN_TYPE).read_file()
test_path = read_config.ReadConfig(TEST_TYPE).read_file()
# 文件存储路径
root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\"
# 读取参数
x_train,y_train,x_valid,y_valid = pickle.load(open(train_path,'rb')) 

x_train = []
for data in open(root + "trian.txt"):
    x_train.append(data)
x_valid = []
for data in open(root + "valid.txt"):
    x_valid.append(data)

x_test = ""
y_test = ""
if test_path == "":
    pass
else:   
    x_test,y_test = pickle.load(open(test_path,'rb'))


""" 生成数据字典 """
def init_dict(data_list):
    # 调用字典转换方法
    word_dict,participle_arr = participle_to_label(data_list) 
    f = open(root + "dict_txt_all.txt","w")
    f.write(str(word_dict))
    # 打印字典长度（验证下结果是否正确）
    print("the dataset length is :%d"%len(data_list))

"""生成训练文本集合"""
def create_list(docList, fileName, dicts):
    word_dict,participle_arr = participle_to_label(docList)
    tf_method(dicts,participle_arr,fileName,data_path=root)

"""创建验证集合文本"""
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

    run_type = input('请输入运行模式:1，训练模式 2，模型验证模式:')

    data_list = list(x_train)
    # 将验证集合代码加入list里面
    for validStr in x_valid:
        data_list.append(validStr)
    # 将测试集合的代码加入List里面

    if run_type == '1':
        # pass
        init_dict(data_list)
    elif run_type == '2':
        # 将x_test写入文本中
        for testStr in x_test:
            data_list.append(testStr)
        create_test_test(x_test)
        init_dict(data_list)
    else:
        print("请按照规则选择运行模式")

    # 生成数据字典
    
    f = open(root + "dict_txt_all.txt","r",encoding="utf-8")
    dicts = f.read()
    dictinfo = eval(dicts)

    train_dict_path = "课程设计\\code\\Rnn\\datasets\\train_list.txt"
    valid_dict_path = "课程设计\\code\\Rnn\\datasets\\valid_list.txt"

    # 生成训练集合
    create_list(list(x_train),"train_list.txt",dictinfo)


    files1 = open(root+"train_list_end.txt", "w",encoding="utf-8")
   

    i = 0
    counts = 0
    train_list = []
    for files in open(train_dict_path):
        label = str(y_train[i])
        if int(label) == 0:
            for x in range(9):
                list1 = files.strip("\n").split()
                random.shuffle(list1)
                strClass = ",".join(list1)
                str2 = strClass + "\t" + str(int(label)) + "\n"
                train_list.append(str2)
                print("step i is :%s" % i)
                x+=1
        elif int(label) == 1:
            for x in range(4):
                list1 = files.strip("\n").split()
                random.shuffle(list1)
                strClass = ",".join(list1)
                str2 = strClass + "\t" + str(int(label)) + "\n"
                train_list.append(str2)
                print("step i is :%s" % i)
                x+=1
        elif int(label) == 2:
            for x in range(4):
                list1 = files.strip("\n").split()
                random.shuffle(list1)
                strClass = ",".join(list1)
                str2 = strClass + "\t" + str(int(label)) + "\n"
                train_list.append(str2)
                print("step i is :%s" % i)
                x+=1
        elif int(label) == 4:
            for x in range(4):
                list1 = files.strip("\n").split()
                random.shuffle(list1)
                strClass = ",".join(list1)
                str2 = strClass + "\t" + str(int(label)) + "\n"
                train_list.append(str2)
                print("step i is :%s" % i)
                x+=1
        elif int(label) == 6:
            for x in range(2):
                list1 = files.strip("\n").split()
                random.shuffle(list1)
                strClass = ",".join(list1)
                str2 = strClass + "\t" + str(int(label)) + "\n"
                train_list.append(str2)
                print("step i is :%s" % i)
                x+=1
        elif int(label) == 7:
            for x in range(12):
                list1 = files.strip("\n").split()
                random.shuffle(list1)
                strClass = ",".join(list1)
                str2 = strClass + "\t" + str(int(label)) + "\n"
                train_list.append(str2)
                print("step i is :%s" % i)
                x+=1
        else:
            list1 = files.strip("\n").split()
            random.shuffle(list1)
            strClass = ",".join(list1)
            str2 = strClass + "\t" + str(int(label)) + "\n"
            train_list.append(str2)
            print("step i is :%s" % i)
        counts+=1
        i+=1
    # 打乱读入顺序
    random.shuffle(train_list)
    for data in train_list:
        files1.write(data)  
    files2 = open(root+"valid_list_end.txt", "w",encoding="utf-8")

    print("trian_file 生成完成")

    is_continue = input("是否继续操作, 1继续操作  2结束程序 ")

    if is_continue == '2':
        exit(0)
    else:
        # 生成测试集合
        create_list(list(x_valid),"valid_list.txt",dictinfo) 

        j = 0
        counts = 0
        valid_list = []
        for files in open(valid_dict_path):
            label = str(y_valid[j])
            strClass = ",".join(files.strip("\n").split())
            str2 = strClass + "\t" + str(int(label)) + "\n"
            valid_list.append(str2)
            print("step j is :%s" % j)
            counts+=1
            j+=1
        # 打乱读入顺序
        random.shuffle(valid_list)
        for data in valid_list:
            files2.write(data)  