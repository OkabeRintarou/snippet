#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn

def gen_data():
    num_inputs = 2
    num_examples = 1000

    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    return (features, labels)

def data_iter(batch_size, features, labels):
    dataset = gdata.ArrayDataset(features, labels)
    return gdata.DataLoader(dataset, batch_size, shuffle=True)

def training():
    features, labels = gen_data()
    lr = 0.03
    num_epochs = 3
    batch_size = 10

    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))

    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_l = loss(net(features), labels)
        print('epoch %d, loss %f' % (epoch, train_l.mean().asnumpy()))
