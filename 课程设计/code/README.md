# 大数据课程设计代码操作文档

代码目录

1. ![](..\code\image\code_list.png)

bilstm_net.py  bi-lstm 神经网络模型

create_data.py  创建预训数据。

create_dict.py  创建字典

infer_test.py   预测代码

net.py 网络代码

output.py 输出代码

text_reader.py 读取数据的方法

train.py 训练方法

valid_model_acc.py  验证模型正确率方法

valid.py 验证方法



至于如何操作

1. 运行create_dict.py 生成字典和数据
2. 运行trian.py训练代码
3. 运行infer_test预测代码并输出
4. 运行valid_model_acc验证准确率



注意：

操作create_dict的时候，需要自己做一些事情。

![](.\image\create_dict_1.png)

先运行代码段1 生成dict

再运行代码段2 生成训练集

![](..\code\image\create_dict_2.png)

最后运行代码段2