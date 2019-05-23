#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT


from sklearn.tree import DecisionTreeClassifier
import ReadData
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
# 画图
import matplotlib as mpl
import matplotlib.pyplot as plt

writpath = "E:\\PythonProject\\大数据处理与实践\\课程设计\\res\\res.txt"

x_train,y_train,x_valid,y_valid = ReadData.readFile()
#文本特征向量化
vec = CountVectorizer()
X_train1 = vec.fit_transform(x_train)
X_test = vec.transform(x_valid)

list1 = dict()
range1 = []
# 训练模型，限制树的最大深度i
f = open(writpath,"w")
# step 302 max value 0.615
for i in range(0, 450):
    clf = DecisionTreeClassifier(max_depth=(i+1))
    #拟合模型
    clf.fit(X_train1, y_train)
    Z = clf.predict(X_test)
    list1[i] = (sum(Z == y_valid) / len(y_valid))
    range1.append(i+1)
    f.write(str("step "+ str(i+1) + " " + str(sum(Z == y_valid) / len(y_valid))) + "\n")
    print("step "+ str(i+1) + " the accurary is " + str(sum(Z == y_valid) / len(y_valid)))
    res = str("now the max value:" + str(max(list1.values())))
    print(res)

print()
res = str("the max value:" + str(max(list1.values())))
f.write(res)
print(res)
plt.plot(list1.items(), list1.values(), color='blue', label='ID3 training accuracy')
plt.xlabel(' times')
plt.ylabel('ID3 accurary')
plt.show()

