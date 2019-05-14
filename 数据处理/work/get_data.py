#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 统计每个人访问该地区的频率（分为，访问每一张图片的频率，每一类别的频率）
import json
from threading import Thread
import threading
from time import sleep, ctime 
import time
import matplotlib.pyplot as plt

areaDict = dict()
areaDict['001'] = 'Residential area'
areaDict['002'] = 'School'
areaDict['003'] = 'Industrial park'
areaDict['004'] = 'Railway station'
areaDict['005'] = 'Airport'
areaDict['006'] = 'Park'
areaDict['007'] = 'Shopping area'
areaDict['008'] = 'Administrative district'
areaDict['009'] = 'Hospital'

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print ("开启线程： " + self.name)
        # 获取锁，用于线程同步
        threadLock.acquire()
        print_time(self.name, self.counter, 3)
        handle()
        # 释放锁，开启下一个线程
        threadLock.release()

def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

def handle():
    filePath = "D:\\SpringBoot\\workspace\\big_data_handle_out\\train_user_result5.txt"
    visitList = []
    for fileName in open(filePath):
        visitList.append(fileName)
    i = 0
    for visit in visitList:
        print("正在处理第" + str(i) + "组数据")
        strs = json.loads(visit)
        dateData = strs['visitTime']
        dateData1 = sorted(dateData)
        dateDict = dict()
        for data in dateData1:
            dateDict[data] = dateData[data]
                        # 绘制图像
        fileName = strs['fileName']
        classNo = str(fileName).split("_")[1].split(".")[0]
        x1 = []
        y1 = []
        plt.cla()
        for d,x in dateDict.items():
            x1.append(d)
            y1.append(x)  
        plt.xticks(rotation = 0)
        plt.plot(x1, y1)
        plt.xlabel(u'date')
        plt.ylabel(u'visitTimes')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(fileName + " the class no is " + areaDict[classNo])
        plt.savefig("E:\\dataSet\\初赛赛题\\train_handle\\"+fileName+".jpg")
        print("保存图片路径为：E:\\dataSet\\初赛赛题\\train_handle\\"+fileName+".jpg")
        print("第" + str(i) + "组数据，处理完成！")
        i+=1
        # plt.show()


threadLock = threading.Lock()
threads = []

# 创建新线程
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)


if __name__ == "__main__":
    # 开启新线程
    thread1.start()
    thread2.start()

    # 添加线程到线程列表
    threads.append(thread1)
    threads.append(thread2)

    # 等待所有线程完成
    for t in threads:
        t.join()
    print ("退出主线程")