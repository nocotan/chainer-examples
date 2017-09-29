# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, reporter
from chainer import Link, Chain, ChainList
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from scipy.misc import imsave

class Generator(Chain):
    def __init__(self, z_dim):
        super(Generator, self).__init__(
            l1=L.Deconvolution2D(z_dim, 128, 3, 2, 0),
            bn1=L.BatchNormalization(128),
            l2=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn2=L.BatchNormalization(128),
            l3=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn3=L.BatchNormalization(128),
            l4=L.Deconvolution2D(128, 128, 3, 2, 2),
            bn4=L.BatchNormalization(128),
            l5=L.Deconvolution2D(128, 1, 3, 2, 2, outsize=(28, 28)),
        )
        self.train = True

    def __call_(self, z):
        h = self.bn1(F.relu(self.l1(z)), test=not self.train)
        h = self.bn2(F.relu(self.l2(h)), test=not self.train)
        h = self.bn3(F.relu(self.l3(h)), test=not self.train)
        h = self.bn4(F.relu(self.l4(h)), test=not self.train)
        x = F.sigmoid(self.l5(h))
        return x

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            l1=L.Convolution2D(None, 32, 3, 2, 1),
            bn1=L.BatchNormalization(32),
            l2=L.Convolution2D(None, 32, 3, 2, 2),
            bn2=L.BatchNormalization(32),
            l3=L.Convolution2D(None, 32, 3, 2, 1),
            bn3=L.BatchNormalization(32),
            l4=L.Convolution2D(None, 32, 3, 2, 1),
            bn4=L.BatchNormalization(32),
            l5=L.Convolution2D(None, 1, 3, 2, 1),
        )
        self.train = True

    def __call__(self, x):
        h = self.bn1(F.leaky_relu(self.l1(x)), test=not self.train)
        h = self.bn2(F.leaky_relu(self.l2(h)), test=not self.train)
        h = self.bn3(F.leaky_relu(self.l3(h)), test=not self.train)
        h = self.bn4(F.leaky_relu(self.l4(h)), test=not self.train)
        y = self.l5(h)
        return y
