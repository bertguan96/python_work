import os
import pickle
import numpy as np



data_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\data\\data_student.pkl"
x_train,y_train,x_valid,y_valid = pickle.load(open(data_path,'rb')) 


root = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\"

# 分词过后的数组
participle_arr = []
# 词向量最终结果
doc_term_matrix = []

def participle_to_label(text):
    # 词向量
    word_dict = dict()
    word_set = set()
    for doc in text:
        doc_arr = doc.split()
        origin_arr = []
        participle_arr.append(doc_arr)
    # 将结果转换为set
    for i in range(len(participle_arr)):
        word_set = word_set.union(participle_arr[i])
    # word_set = set(participle_arr[0]).union(participle_arr[1]).union(participle_arr[2])
    # 将set转换为dict
    i = 0
    word_dict = dict()
    for word in word_set:
        word_dict[word] = i
        i+=1
    return word_dict,participle_arr

"""
    将participle_arr数组中的数据转换为对应dict中的位置
"""
def tf(word_dict, participle_arr):       
    index = 0
    index_arr = []
    for participle_word in participle_arr:
        index_list = []
        for word in participle_word:
            index = word_dict[word]
            index_list.append(index)
        participle_word = []
        participle_word = index_list
        index_arr.append(participle_word)
    participle_arr = []
    participle_arr = index_arr
    return participle_arr

""" 
    @method create file list
    @param data_path the files root path
    @param text_list need to write path
    @param fileName create files name
"""
def create_train_text(data_path, text_list,fileName):
    if os.path.exists(data_path):
        files = open(data_path+fileName, "w",encoding="utf-8")
        for text in text_list:
            files.write(text + "\n")
    else:
        os.makedirs(data_path)

""" make the word to vec """
def read_file_word_2_vec(docList, fileName, data_path):
    word_dict,participle_arr = participle_to_label(docList)
    f = open("E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\dict1_txt.txt","w")
    f.write(str(word_dict))
    tf_method(word_dict,participle_arr,fileName,data_path)
        

def tf_method(word_dict, participle_arr,fileName,data_path):
    participle_arr = tf(word_dict,participle_arr)
    # 计算词向量
    files = open(data_path+fileName, "w",encoding="utf-8")
    for participle_word in participle_arr:
        str1 = ""
        for i in participle_word:
            str1 += ","+str(i)
        str1 = str1.strip(",")
        files.write(str1+ "\n")
    files.flush()

def create_dict(fileName,data_path,filePath):
    docList = []
    for doc in open(filePath):
        docList.append(doc)
    read_file_word_2_vec(docList=docList, fileName=fileName, data_path=data_path)

def write_classifiy(strs, label, data_path, files):
    strClass = ",".join(strs.strip("\n").split())
    str2 = strClass + "\t" + str(label)
    
    files.write(str2)  


# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])

    return len(line.keys())

if __name__ == "__main__":
    data_path = root + "datasets\\"
    # create_train_text(data_path,x_train, "train_list.txt")
    # create_train_text(data_path,x_valid, "valid_list.txt")

    # train_list_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\train_list.txt"
    valid_list_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\valid_list.txt"
    # create_dict(fileName="trian_dict.txt",data_path=data_path,filePath=train_list_path)
    create_dict(fileName="valid_dict.txt",data_path=data_path,filePath=valid_list_path)

    # train_dict_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\trian_dict.txt"
    valid_dict_path = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\valid_dict.txt"
    # files1 = open(data_path+"trian_dict1.txt", "w",encoding="utf-8")
    files2 = open(data_path+"valid_dict1.txt", "w",encoding="utf-8")

    # i = 0
    # for files in open(train_dict_path):
    #     label = str(y_train[i])
    #     strClass = ",".join(files.strip("\n").split())
    #     str2 = strClass + "\t" + str(label) + "\n"
    #     files1.write(str2)  
    #     print("step i is :%s" % i)
    #     i+=1

    j = 0
    for files in open(valid_dict_path):
        label = str(y_valid[j])
        strClass = ",".join(files.strip("\n").split())
        str2 = strClass + "\t" + str(label) + "\n"
        files2.write(str2)  
        print("step j is :%s" % j)
        j+=1