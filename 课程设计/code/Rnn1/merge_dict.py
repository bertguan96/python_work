

path1 = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\"

path2 = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\predatasets\\"

dict1 = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\datasets\\dict_txt_all.txt"

dict2 = "E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\Rnn\\predatasets\\dict_txt_all.txt"

def read_dict(path):
    f = open(path,"r",encoding="utf-8")
    dicts = f.read()
    dictinfo = eval(dicts)
    return dictinfo

if __name__ == "__main__":
    dictinfo1 = read_dict(dict1)
    dictinfo2 = read_dict(dict2)
    end_dict = dict(dictinfo1)
    end_dict.update(dict(dictinfo2))
    f = open(path1 + "dict_all.txt","w",encoding="utf-8")
    f.write(str(end_dict))
    f = open(path2 + "dict_all.txt","w",encoding="utf-8")
    f.write(str(end_dict))