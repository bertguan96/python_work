import numpy as np
import paddle.fluid as fluid

# 创建执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 保存预测模型路径
save_path = 'E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\cnn\\infer_model\\'
# 从模型中获取预测程序、输入数据名称列表、分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


# 获取数据
def get_data(sentence):
    # 读取数据字典
    with open('E:\\PythonProject\\大数据处理与实践\\课程设计\\code\\cnn\\datasets\\dict_txt1.txt', 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])
    dict_txt = dict(dict_txt)
    # 把字符串数据转换成列表数据
    keys = dict_txt.keys()
    print(keys)
    data = []
    strs_list = str(sentence).split()
    print(strs_list)
    for s in strs_list:
        # 判断是否存在未知字符
        print(s)
        if not s in keys:
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data


data = []
# 获取图片数据
data1 = get_data('gbh stanleypark avianexcellence')
data2 = get_data('bookfotografico nataliaabramova ritratto portrait ragazze model modella closeup canon villadoriapamphilli')
data.append(data1)
data.append(data2)

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]

# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 分类名称
names = ['民生', '文化', '娱乐', '体育', '财经',
         '房产', '汽车', '教育', '科技']

# 获取结果概率最大的label
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))