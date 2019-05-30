import paddle.fluid as fluid

def textcnn_net_v1(data,
            label,
            dict_dim,
            emb_dim=156,
            hid_dim=156,
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
            act="relu",
            pool_type="max")
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(input=[convs_out], size=hid_dim2, act="tanh")
    # add dropout to
    dropout_1 = fluid.layers.dropout(x=fc_1,dropout_prob=0.2,name="dropout")
    
    # v1 新增一层dropt层
    fc_2 = fluid.layers.fc(input=[dropout_1], size=hid_dim2, act="tanh")

    dropout_2 = fluid.layers.dropout(x=fc_2,dropout_prob=0.2,name="dropout")
    # softmax layer
    fc_3 = fluid.layers.fc(input=[dropout_2], size=96, act="tanh")

    prediction = fluid.layers.fc(input=[fc_3], size=class_dim, act="softmax")

    return prediction

def textcnn_net_v2(data,
            label,
            dict_dim,
            emb_dim=192,
            hid_dim=156,
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
            act="relu",
            pool_type="max")
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(input=[convs_out], size=hid_dim2, act="tanh")
    # add dropout to
    dropout_1 = fluid.layers.dropout(x=fc_1,dropout_prob=0.5,name="dropout")
    
    # v1 新增一层dropt层
    fc_2 = fluid.layers.fc(input=[dropout_1], size=hid_dim2, act="tanh")

    dropout_2 = fluid.layers.dropout(x=fc_2,dropout_prob=0.4,name="dropout")
    # softmax layer
    fc_3 = fluid.layers.fc(input=[dropout_2], size=96, act="tanh")

    dropout_3 = fluid.layers.dropout(x=fc_3,dropout_prob=0.4,name="dropout")

    prediction = fluid.layers.fc(input=[dropout_3], size=class_dim, act="softmax")

    return prediction



def textcnn_net_v3(data,
            label,
            dict_dim,
            emb_dim=192,
            hid_dim=192,
            hid_dim2=156,
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
    # add dropout to
    dropout_1 = fluid.layers.dropout(x=fc_1,dropout_prob=0.5,name="dropout")
    
    # v1 新增一层dropt层
    fc_2 = fluid.layers.fc(input=[dropout_1], size=hid_dim2, act="tanh")

    dropout_2 = fluid.layers.dropout(x=fc_2,dropout_prob=0.4,name="dropout")
    # softmax layer
    fc_3 = fluid.layers.fc(input=[dropout_2], size=156, act="tanh")

    dropout_3 = fluid.layers.dropout(x=fc_3,dropout_prob=0.3,name="dropout")

    fc_4 = fluid.layers.fc(input=[dropout_3], size=156, act="tanh")

    prediction = fluid.layers.fc(input=[fc_4], size=class_dim, act="softmax")

    return prediction