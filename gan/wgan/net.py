# -*- coding: utf-8 -*-
import chainer
from chainer import functions as F
from chainr import links as L


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.dc1 = L.Deconvolution2D(None, 256, 4, stride=1, pad=0, nobias=True)
            self.dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, nobias=True)
            self.dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, nobias=True)
            self.dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)

    def __call__(self, z):
        h = F.reshape(z, (z.shape[0], -1, 1, 1))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.tanh(self.dc4(h))
        return h


class Critic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, 64, 6, stride=2, pad=1, nobias=True)
            self.c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, nobias=True)
            self.c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, nobias=True)
            self.c3 = L.Convolution2D(256, 1, 4, stride=1, pad=0, nobias=True)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)

    def clamp(self, lower=0.01, upper=0.01):
        for params in self.params():
            params_clipped = F.clip(params, lower, upper)
            params.data = params_clipped.data

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = self.c3(h)
        h = F.sum(h) / h.size
        return h
