#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\cnn\\result\\training_output.txt"

resultDict =dict()
j = 0
for i in open(root):
    resultDict[j] = str(i).strip('\n')
    j+=1
# 画图
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.figure("RNN result")
plt.title("RNN accuracy result")
plt.plot(list(resultDict.keys()), list(resultDict.values()), color='blue', label='KNN training accuracy')
plt.xlabel('times')
plt.ylabel('RNN accuracy')
plt.show()