#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common import d2lzh as d2l
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import loss as gloss, nn

def load_data():
    batch_size=256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    return train_iter, test_iter

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n

def training(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()

            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc: %.3f'
                % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

def train_v0():
    num_inputs, num_outputs = 784, 10
    num_epochs, lr = 5, 0.1
    batch_size = 256

    train_iter, test_iter = load_data()
    W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
    b = nd.zeros(num_outputs)
    W.attach_grad()
    b.attach_grad()
    net = lambda X: softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

    training(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr);

def train_v1():
    num_inputs, num_outputs = 784, 10
    batch_size = 256
    train_iter, test_iter = load_data()
    net = nn.Sequential()
    net.add(nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})
    num_epochs = 5

    training(net, train_iter, test_iter, loss , num_epochs, batch_size, None, None, trainer);
