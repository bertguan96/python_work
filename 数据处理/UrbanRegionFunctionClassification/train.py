
import paddle.fluid as fluid
import model
import paddle as paddle
import reader

crop_size = 224
resize_size = 250

model = model.MultiModel()
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model.prediction, label=model.label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model.prediction, label=model.label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)

# 定义一个使用GPU的执行器
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[model.image, model.visit,model.one_hot])

# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader('images/train.list', crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader('images/test.list', crop_size), batch_size=32)
