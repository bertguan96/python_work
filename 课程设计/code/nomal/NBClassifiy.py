from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import BernoulliNB 
from sklearn.metrics import accuracy_score
import ReadData

x_train,y_train,x_valid,y_valid = ReadData.readFile()

#文本特征向量化
vec = CountVectorizer()
X_train1 = vec.fit_transform(x_train)
X_test = vec.transform(x_valid)

#3.使用朴素贝叶斯进行训练
# mnb = MultinomialNB() 0.6
mnb = BernoulliNB()  #  0.409
# mnb = GaussianNB()   # 使用默认配置初始化朴素贝叶斯 0.51
mnb.fit(X_train1.toarray(),y_train)    # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(X_test.toarray())     # 对参数进行预测


print(accuracy_score(y_predict, y_valid))
