#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT



import json
import os

# 获取当前文件的上一级的绝对路径
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 配置文件路径
config = dir_path + "/config/config.json"

# 获取配置文件中的训练集路径
def get_data_path():
    f = open(config,encoding="utf-8").read()
    f = json.loads(f)['trainData']  
    return f
# 生成基础路径
def create_file_list():
    f = get_data_path()
    file_name_list = os.listdir(f)
    f_save = open(dir_path + "/data/file_list.txt","w",encoding="utf-8")
    for fileName in file_name_list:
        file_path = f + "\\" + fileName
        f_save.write(file_path + "\n")

def read_file(path):
    f = open(path,encoding="utf-8")
    list11 = f.readlines()
    print(list11[5].split())


read_file_list = []
for read in open(dir_path + "/data/file_list.txt","r",encoding="utf-8"):
    read_file_list.append(str(read).strip("\n"))
read_file(read_file_list[0])
