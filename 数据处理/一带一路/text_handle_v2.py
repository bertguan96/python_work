#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 统计每个人访问该地区的频率（分为，访问每一张图片的频率，每一类别的频率）

import os

"""
    统计类别如下
    001
    002
    003
    004
    005
    006
    007
    008
    009
"""

file_dir = "E:\\dataSet\\初赛赛题\\train_visit\\train\\"
files = os.listdir(file_dir)
# 得到所有的文件路径
fileNames = [ file_dir+fileName for fileName in files]
a = open('filePath.txt','w',encoding='UTF8')
for fileName in fileNames:
    a.write(fileName + "\n")
