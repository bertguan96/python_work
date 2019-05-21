#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT


from sklearn.tree import DecisionTreeClassifier
import ReadData
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块


writpath = "E:\\PythonProject\\大数据处理与实践\\课程设计\\res\\res.txt"

x_train,y_train,x_valid,y_valid = ReadData.readFile()
#文本特征向量化
vec = CountVectorizer()
X_train1 = vec.fit_transform(x_train)
X_test = vec.transform(x_valid)

list1 = []
range1 = []
# 训练模型，限制树的最大深度4
f = open(writpath,"w")
for i in range(250, 1000):
    clf = DecisionTreeClassifier(max_depth=(i+1))
    #拟合模型
    clf.fit(X_train1, y_train)
    Z = clf.predict(X_test)
    list1.append(sum(Z == y_valid) / len(y_valid))
    range1.append(i+1)
    f.write(str("step "+ str(i+1) + " " + str(sum(Z == y_valid) / len(y_valid))) + "\n")
    print("step "+ str(i+1) + " " + str(sum(Z == y_valid) / len(y_valid)))
print()
f.write(str("the max value:" + max(list1)))
print(str("the max value:" + max(list1)))