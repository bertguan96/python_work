#!/usr/bin/env python
#-*- coding:utf8 -*-
# Power by GJT

import paddle.fluid as fluid

"""
    max acc 0.653

"""
def cnn_net(data,
            label,
            dict_dim,
            class_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            win_size=3,
            is_infer=False):
    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    # convolution layer
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="tanh",
        pool_type="max")
    # full connect layer
    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)
    batch_normal = fluid.layers.batch_norm(input=fc_1)
    # softmax layer
    prediction = fluid.layers.fc(input=[batch_normal], size=class_dim, act="softmax")

    return prediction


def textcnn_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=10,
            win_sizes=None,
            is_infer=False):
    """
    Textcnn_net
    """
    if win_sizes is None:
        win_sizes = [1, 2, 3]

    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # convolution layer
    convs = []
    for win_size in win_sizes:
        conv_h = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max")
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(input=[convs_out], size=hid_dim2, act="tanh")
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")

    return prediction