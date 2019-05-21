#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT
# file 模型文件

import paddle.fluid as fluid
import numpy as np 
import paddle as paddle
from PIL import Image
import os

class MultiModel(object):
    def __init__(self):
        self.image = fluid.layers.data(name='image', shape=[3,88,88],dtype='float32')
        self.visit = fluid.layers.data(name='visit', shape=[7,26,24],dtype='float32')

        self.label = fluid.layers.data(name='label', shape=[1], dtype='int32')
        self.on_hot = fluid.layers.one_hot(input=self.label, depth=9)

        self.global_step = fluid.layers.create_parameter(name="global_step",shape=[1],dtype='float32')
        self.training = fluid.layers.data(dtype='bool',name='training',shape=[1])
        self.output_image = self.image_network(self.image,'Resnet-image')
        print(self.output_image)
        self.output_visit = self.visit_network(self.visit, 'Resnet-visit')
        print(self.output_visit)
        concat_list = [self.output_image,self.output_visit]
        self.output = fluid.layers.concat(input=concat_list,  axis=1)
        self.prediction = fluid.layers.fc(self.output,size=9)

        self.loss = self.get_loss(self.prediction, self.one_hot)  
        self.batch_size = 512
    
        self.correct_prediction = fluid.layers.equal('correct_prediction',fluid.layers.argmax(self.prediction, 1) == fluid.layers.argmax(self.one_hot, 1)) 
        self.accuracy = fluid.layers.reduce_mean('accuracy',fluid.layers.cast(correct_prediction, dtype="float32"))
            # tf.summary.scalar('train_accuracy_concat', self.accuracy)
        print(self.accuracy)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        # self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss, global_step=self.global_step)
        self.optimizer = fluid.optimizer.AdamOptimizer(1e-3,1e-06,fluid.regularizer.L2DecayRegularizer,"optimizer",0.0004)
        # self.merged = tf.summary.merge_all()
        
        print("网络初始化成功")

    """
        卷积网络
        @param x 卷积参数
        @param input_filters 输入步长
        @param ouput_filters 输出步长
        @param kernel 核心
        @param 步长
    """
    def conv2d(self, x, input_filters, output_filters, kernel, strides=1):
        shape = [kernel, kernel, input_filters, output_filters]
        # 采用高斯分布初始化器
        weight = fluid.layers.create_parameter(shape=shape,
                                               default_initializer=fluid.initializer.TruncatedNormal(scale=0.1),
                                               name='weight',
                                               dtype='float32')
    
        return fluid.layers.conv2d(input=x,num_filters=kernel,filter_size=2,stride=[strides,strides],padding=[0,0],param_attr='weight')

    """
        卷积网络
    """
    def residual(self, x, num_filters, strides, name, with_shortcut=False):
        conv1 = self.conv2d(x, num_filters[0], num_filters[1], kernel=1, strides=strides)
        bn1 = fluid.layers.batch_norm(input=conv1, param_attr='training')
        relu1 = fluid.layers.relu(bn1)
        conv2 = self.conv2d(relu1, num_filters[1], num_filters[2], kernel=3)
        bn2 =  fluid.layers.batch_norm(input=conv2,param_attr='training')
        relu2 = fluid.layers.relu(bn2)
        conv3 = self.conv2d(bn2, num_filters[2], num_filters[3], kernel=1)
        bn3 =  fluid.layers.batch_norm(input=conv3,param_attr='training')
        if with_shortcut:
            shortcut = self.conv2d(x, num_filters[0], num_filters[3], kernel=1, strides=strides)
            bn_shortcut = fluid.layers.batch_norm(input=conv3,param_attr='training')
            residual = fluid.layers.relu(bn_shortcut + bn3, name)
        else:
            residual = fluid.layers.relu(x + bn3, name)
        return residual

    """
        图片网络
        @param image 图片
        @network Resnet-image
    """
    def image_network(self, image, network):
        channel = 16
        if network == "Resnet-image":
            conv = self.conv2d(image, 3, channel, 7, 1)
            bn = fluid.layers.batch_norm(input=conv, param_attr='training')
            relu = fluid.layers.relu(bn,'stage1')
            pool = fluid.layers.pool2d(relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
            res = self.residual(pool, [channel, channel//2, channel//2, channel*2], 1,'stafe2' ,with_shortcut=True)
            res = self.residual(res, [channel*2, channel, channel, channel*4], 2,'stage3' ,with_shortcut=True)
            res = self.residual(res, [channel*4, channel*2, channel*2, channel*8], 2,'stage4' ,with_shortcut=True)
            res = self.residual(res, [channel*8, channel*4, channel*4, channel*16], 2, with_shortcut=True)
            pool = fluid.layers.pool2d(res, pool_size=[1, 6, 6, 1], pool_type='avg', pool_padding=[1,1], pool_stride=[1, 1, 1, 1],name='stage5')
            flatten = fluid.layers.flatten('fc',pool, axis=3)
        return flatten

    """
        访问数据网络
        @image image 图片
        @network Resnet-visit
    """
    def visit_network(self, image, network):
        channel = 32
        if network == "Resnet-visit":
            conv = self.conv2d(image, 7, channel, 7, 1)
            bn = fluid.layers.batch_norm(input=conv,param_attr='training')
            relu = fluid.layers.relu(bn,'stage1')
            res = self.residual(relu, [channel, channel//2, channel//2, channel*2], 1,'stafe2' ,with_shortcut=True)
            res = self.residual(res, [channel*2, channel, channel, channel*4], 2,'stage3' ,with_shortcut=True)
            res = self.residual(res, [channel*4, channel*2, channel*2, channel*8], 2,'stage4' ,with_shortcut=True)
            res = self.residual(res, [channel*8, channel*4, channel*4, channel*16], 2, with_shortcut=True)
            pool = fluid.layers.pool2d(res, pool_size=[1, 1, 4, 1], pool_type='avg', pool_padding=[1], pool_stride=[1, 1, 1, 1], name='stage5')
            flatten = fluid.layers.flatten('fc', pool, axis=3)
        return flatten

    """
        获取损失值
    """
    def get_loss(self, output_concat, onhot):
        if self.stage == "loss":
            losses = fluid.layers.sigmoid_cross_entropy_with_logits(x=output_concat, label=onhot)
            loss = fluid.layers.reduce_mean(losses)
        return loss


if __name__ == "__main__":
    model = MultiModel()
    print(model)

    