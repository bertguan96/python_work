#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 创建数据字典
import os
import numpy as np
from create_data_v1 import participle_to_label
from create_data_v1 import tf_method
import json
import utils
import pickle


# 设置读取类型
TRIAN_TYPE = "train"
TEST_TYPE = "test"

# 文件存储路径
root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\"

# predataset 
predataset = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\predatasets\\"

data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 


""" 生成数据字典 """
def init_dict(data_list):
    # 调用字典转换方法
    word_dict,participle_arr = participle_to_label(data_list) 
    f = open(predataset + "dict_all.txt","w",encoding="utf-8")
    f.write(str(word_dict))
    # 打印字典长度（验证下结果是否正确）
    print("the dataset length is :%d"%len(data_list))

"""生成训练文本集合"""
def create_list(docList, fileName, dicts):
    word_dict,participle_arr = participle_to_label(docList)
    tf_method(dicts,participle_arr,fileName,data_path=predataset)





if __name__ == "__main__":
    # # 完成数据初始化操作
    # instance = root + "StackOverflow.txt"
    # label = root + "StackOverflow_gnd.txt"
    # datas = utils.load_file(instance)
    # datas1 = utils.delete_punctuation(datas)
    # utils.save_file(datas1,root + "StackOverflow_all.txt")
    # datas = utils.load_file(root + "StackOverflow_all.txt")
    # utils.divide_data_set(datas,0.8,root +"StackOverflow")
    # datas_label = utils.load_file(label)
    # utils.divide_data_set(datas_label,0.8,root +"StackOverflow_label")

    # 生成字典(2w)
    # datas2 = utils.load_file(root + "StackOverflow_all.txt")
    # init_dict(datas2)
    
    # 读取训练集和
    """"附加数据"""
    # 训练集合数据
    x_train1 =  utils.load_file(root + "StackOverflow_train.txt")
    # 训练集合标签
    x_label1 = utils.load_file(root + "StackOverflow_label_train.txt")
    # 测试集合数据
    y_train1 = utils.load_file(root + "StackOverflow_valid.txt")
    # 测试集合变迁
    y_label1 = utils.load_file(root + "StackOverflow_label_valid.txt")
    # 数据清洗（把label<1）
    x_train1,x_label1,y_train1,y_label1 = utils.create_data(x_train1,x_label1,y_train1,y_label1)

    """"附加数据"""
    """"老师给的原始数据"""
    x_train2 = list(x_train)
    y_train2 = list(y_train)

    x_valid2 = list(x_valid)
    y_valid2 = list(y_valid)
    """"老师给的原始数据"""
    for trian in x_train1:
        x_train2.append(trian)

    for label in x_label1:
        y_train2.append(label)

    for trian1 in y_train1:
        x_valid2.append(trian1) 

    for label1 in y_label1:
        y_valid2.append(label1)

    # docList1 = []
    # for data in x_train2:
    #     docList1.append(data)
    # for data in x_valid2:
    #     docList1.append(data)

    # init_dict(docList1)

    # 读取字典
    f = open(predataset + "dict_all.txt","r",encoding="utf-8")
    dicts = f.read()
    dictinfo = eval(dicts)

    train_dict_path = predataset + "train_list.txt"
    valid_dict_path = predataset + "valid_list.txt"

    # 生成训练集合
    create_list(list(x_train2),"train_list.txt",dictinfo)


    files1 = open(predataset+"train_list_end.txt", "w",encoding="utf-8")
   
    print(y_train2[5426])

    i = 0
    counts = 0
    for files in open(train_dict_path):
        label = str(y_train2[i])
        strClass = ",".join(files.strip("\n").split())
        str2 = strClass + "\t" + str(int(label)) + "\n"
        files1.write(str2)  
        print("step i is :%s" % i)
        counts+=1
        i+=1
    files2 = open(predataset+"valid_list_end.txt", "w",encoding="utf-8")

    print("trian_file 生成完成")

    is_continue = input("是否继续操作, 1继续操作  2结束程序 ")

    if is_continue == '2':
        exit(0)
    else:
        # 生成测试集合
        create_list(list(x_valid2),"valid_list.txt",dictinfo) 

        j = 0
        counts = 0
        for files in open(valid_dict_path):
            label = str(y_valid2[j])
            strClass = ",".join(files.strip("\n").split())
            str2 = strClass + "\t" + str(int(label)) + "\n"
            files2.write(str2)  
            print("step j is :%s" % j)
            counts+=1
            j+=1
    

