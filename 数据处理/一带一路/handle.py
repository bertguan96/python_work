
#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 统计每个人访问该地区的频率（分为，访问每一张图片的频率，每一类别的频率）

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