from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import BernoulliNB 
from sklearn.metrics import accuracy_score
import ReadData
# 画图
import matplotlib as mpl
import matplotlib.pyplot as plt
x_train = ReadData.read_file("trian_label.txt")
y_train = ReadData.read_file("trian.txt")
x_valid = ReadData.read_file("valid_label.txt")
y_valid = ReadData.read_file("valid.txt")

#文本特征向量化
vec = CountVectorizer()
X_train1 = vec.fit_transform(x_train)
X_test = vec.transform(x_valid)

#3.使用朴素贝叶斯进行训练
mnb = MultinomialNB()
mnb.fit(X_train1,y_train)    # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(X_test.toarray())     # 对参数进行预测
str(sum(y_predict == y_valid) / len(y_valid))
plt.plot(len(list(y_predict)), list(y_predict), color='blue', label='KNN training accuracy')
plt.xlabel('KNN accuracy')
plt.ylabel('n_neighbors')
plt.show()
