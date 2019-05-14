
#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 统计每个人访问该地区的频率（分为，访问每一张图片的频率，每一类别的频率）

import numpy as np
import json



def getTrainData():
    fileTrain = "E:\\dataSet\\初赛赛题\\train\\train_vector.txt"
    trainList = []
    for train in open(fileTrain):
        newList = []
        list1 = train.split("\n")[0].split(",")
        for a in list1:
            b = int(a)
            newList.append(b)
        trainList.append(newList)
    label = []
    for train in trainList:
        trainList1 = train.pop()
        label.append(trainList1)
    return np.array(trainList),np.array(label)

def getTestData():
    fileTrain = "E:\\dataSet\\初赛赛题\\train\\test_vector.txt"
    testList = []
    for train in open(fileTrain):
        newList = []
        list1 = train.split("\n")[0].split(",")
        for a in list1:
            b = int(a)
            newList.append(b)
        testList.append(newList)
    label = []
    for train in testList:
        testList1 = train.pop()
        label.append(testList1)
    return testList,label



