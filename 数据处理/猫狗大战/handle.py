#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 测试处理文件

import shutil  # 复制文件
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# cat_train_path = "E:\\dataSet\\dog_and_cat\\train\\Cat\\cat.1.jpg"
# img_obj = cv2.imread(cat_train_path)

# im_gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

# cv2.imwrite("canny.jpg", cv2.Canny(im_gray, 200, 300))
# cv2.imshow("canny", cv2.imread("canny.jpg"))
# cv2.waitKey()
# cv2.destroyAllWindows()

# plt.axis("off")
# plt.title("Input Image")
# plt.imshow(median, cmap="gray")

# plt.show()


dayImage = [0]*365
print(dayImage)

# 初始日期，以一年时间计算
# [0,1,0,1,0,1,0,10,1100,1000，……………………，10]
# 365维度,1阶矩阵

# 时间（全局，某一天）
# [0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# 在2点钟访问该地区的人数为3
# 24维度，1阶矩阵


# 总共多少人访问了这个区域
# 尽量用dict，set，list类型
