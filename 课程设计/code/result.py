import matplotlib as mpl
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


resultDict=dict()

resultDict['KNN'] = 0.482
resultDict['决策树'] = 0.612
resultDict['RNNLSTM'] = 0.972
resultDict['朴素贝叶斯'] = 0.612

plt.figure("算法比较输出结果")
plt.title("算法比较输出结果")
plt.bar(list(resultDict.keys()), list(resultDict.values()), color='rgb', label='result')
plt.xlabel('methods')
plt.ylabel('result accuracy')
plt.show()