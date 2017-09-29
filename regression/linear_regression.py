# -*- coding: utf-8 -*-
import chainer
import chainer.links as L


class LinearRegression(chainer.Chain):
    def __init__(self, in_size):
        super(LinearRegression, self).__init__(
            l1=L.Linear(in_size, 1),
        )

    def __call__(self, x):
        return self.l1(x)
