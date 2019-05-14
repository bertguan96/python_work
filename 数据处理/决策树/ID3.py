#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# method 决策树算法sklearn实现

# 导入库
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO
import math

# 测试，没有age时的信息熵
result = -(5/14*math.log2(5/14) + 9/14*math.log2(9/14))
print(result)

#测试，有age时的信息熵
result1 = 5/14*(-3/5*math.log2(3/5) - 2/5*math.log(2/5)) + 4/14*(-4/4*math.log2(4/4)) + 5/14*(-3/5*math.log2(3/5) - 2/5*math.log2(2/5))
print(result1)

# 1 打开测试数据集
allElectronicsData = open(r'E:\\PythonProject\\大数据处理与实践\\数据处理\\决策树\\test.csv','r')
reader = csv.reader(allElectronicsData)
# 读取文件头
headers = next(reader)
#print(headers)

featureList = []
labelList = []

# 2 为每个测试样例，建立一个特征名和值的字典，然后加入到featureList中
for row in reader:
    labelList.append(row[len(row) -1])
    rowDict = {}
    for i in range(1,len(row) -1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
print(labelList)
print(featureList)

# 3 把输入特征集进行转换，例如
'''
{'age': 'youth', 'income': 'high', 'student': 'no', 'credit_rating': 'fair'}
[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
'''
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
#print(type(dummyX))
print(dummyX)
#print(vec.get_feature_names())

# 4 对标签值进行0，1转换
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)

# 5 直接调用库的决策树分类器，entropy表示信息熵
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
#print(clf)

# 6 填入feature_names，还原原有的值，写入test.dot文件
with open("test.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

# 7 此时dumyX为一个n维的向量，取出第一行，修改一下作为一个新的测试数据
# 'age': 'middle_aged', 'income': 'high', 'student': 'no', 'credit_rating': 'fair' 正确结果应该是 买 为1
oneRowX = dummyX[0,:]
print("oneRowX:  ",oneRowX)
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print(newRowX)

# 8 使用新数据进行测试验证
predictedY = clf.predict(newRowX.reshape(1, -1))
print(predictedY)

# 使用graphviz工具，通过命令dot -Tpdf test.dot -o output.pdf生成决策树文档