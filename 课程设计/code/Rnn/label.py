
import pickle
import numpy as np
data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 


import pandas as pd
from imblearn.over_sampling import SMOTE       #过度抽样处理库SMOTE
df=pd.read_table('E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\train_list_end.txt',sep='\t',names=['col1','label'])    
x=df.iloc[:,:-1]
x = list(x['col1'])
x_last = []

for data in x:
    data_list = []
    datas = str(data).split(",")
    for data1 in datas:
        data_list.append(float(data1))
    x_last.append(data_list)
x_last = np.array(x_last)
print(np.shape(x_last))

y=df.iloc[:,-1]
y = list(y)
y_last = []
for data in y:
    y_last.append(list([int(data)]))
y_last = np.array(y_last).reshape(5425,)
import numpy as np
import smote as SMOTE
import matplotlib.pyplot as plt

n_rows = 30

T = np.random.randn(n_rows,2)

c = np.ones((n_rows,1)) # Class
T = np.append(T, c, axis=1)

N = 10
k = 5
smote = SMOTE(T,N,k=k)
synth = smote.over_sampling()
print('# Synth Samps: ', synth.shape[0])


plt.title('SMOTE')
plt.xlabel('Attr 1')
plt.ylabel('Attr 2')
plt.scatter(T[:, 0], T[:, 1], marker='x')
plt.scatter(synth[:, 0], synth[:, 1], marker='x', color='red')
plt.show()