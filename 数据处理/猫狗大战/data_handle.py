#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 数据处理
import shutil  # 复制文件
import os
from PIL import Image
import numpy as np
"""
    0 猫
    1 狗
"""
label = [1,0]
#源地址
dog_origin_path = "E:\\dataSet\\dog_and_cat\\origin\\Dog"
cat_origin_path = "E:\\dataSet\\dog_and_cat\\origin\\Cat"
# 训练地址
dog_train_path = "E:\\dataSet\\dog_and_cat\\train\\Dog"
cat_train_path = "E:\\dataSet\\dog_and_cat\\train\\Cat"
# 测试地址
dog_test_path = "E:\\dataSet\\dog_and_cat\\test\\Dog"
cat_test_path = "E:\\dataSet\\dog_and_cat\\test\\Cat"

"""
    加载模块完成，测试数据和训练数据的划分
"""
def __load_data_set():
    # 读取前400张图片复制到训练集中
    fnames = ['{}.jpg'.format(i) for i in range(400)]
    for fname in fnames:
        src = os.path.join(dog_origin_path, fname)
        dst = os.path.join(dog_train_path, "dog." + fname)
        shutil.copyfile(src, dst)
    for fname in fnames:
        src = os.path.join(cat_origin_path, fname)
        dst = os.path.join(cat_train_path, "cat." + fname)
        shutil.copyfile(src, dst)
    # 读取后200张图片到测试集合中
    fnames = ['{}.jpg'.format(i) for i in range(400,600)]
    for fname in fnames:
        src = os.path.join(dog_origin_path, fname)
        dst = os.path.join(dog_test_path, "dog." + fname)
        shutil.copyfile(src, dst)
    for fname in fnames:
        src = os.path.join(cat_origin_path, fname)
        dst = os.path.join(cat_test_path, "cat." + fname)
        shutil.copyfile(src, dst)
    print('total training cat images:',len(os.listdir(cat_train_path)))
    print('total training dog images:',len(os.listdir(dog_train_path)))
    print('total validation cat images:',len(os.listdir(cat_test_path)))
    print('total validation dog images:',len(os.listdir(dog_test_path)))



"""
    加载文件并转为矩阵向量
    the result type is dict
"""
def __load_file(path_list, label):
    image_list = list()
    i = 0
    for path in path_list:
        img_obj = Image.open(path).convert('L')
        img_obj = img_obj.resize((224, 224), Image.ANTIALIAS)
        image = np.multiply(img_obj, 1.0 / 255.0)
        img_array = np.asarray(image.reshape(1,1,224,224), dtype=np.float32)
        image_tuple = (img_array, label)
        image_list.append(image_tuple)
        i+=1
    return image_list


def load_data():
    # load_data_set()
    cat_train = [os.path.join(cat_train_path, fname) for fname in os.listdir(cat_train_path)]
    dog_train = [os.path.join(dog_train_path, fname) for fname in os.listdir(dog_train_path)]
    cat_image_list = __load_file(cat_train, "cat")
    dog_image_list = __load_file(dog_train, "dog")
    return cat_image_list,dog_image_list

def load_test_data():
    # load_data_set()
    cat_train = [os.path.join(cat_test_path, fname) for fname in os.listdir(cat_test_path)]
    dog_train = [os.path.join(dog_test_path, fname) for fname in os.listdir(dog_test_path)]
    cat_image_list = __load_file(cat_train, "cat")
    dog_image_list = __load_file(dog_train, "dog")
    return cat_image_list,dog_image_list