
from string import punctuation
import re
import nltk


punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'

"""
    加载文件
"""
def load_file(path):
    datas = []
    for data in open(path,encoding="utf-8"):
        datas.append(data)
    return datas

'''
    删除标点符号
'''
def delete_punctuation(fr):
    datas = []
    # 利用正则表达式替换为一个空格
    for line in fr:
        line = re.sub(r"[{}]+".format(punc)," ",line)
        datas.append(line + "\n")
    return datas

"""
    写入文件
"""
def save_file(datas, fileName):
    f = open(fileName,"w",encoding="utf-8")
    for data in datas:
        f.write(data)

"""
    @datas 数据集合
    @size 比例
    @name 存储名称

"""
def divide_data_set(datas,size,name):
    index = len(datas) * size
    save_file(datas[0:int(index)],name + "_train.txt")
    save_file(datas[int(index):],name + "_valid.txt")
