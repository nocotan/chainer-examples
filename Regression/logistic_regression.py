# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L


class LogisticRegression(chainer.Chain):
    def __init__(self, in_size):
        super(LogisticRegression, self).__init__(
            l1=L.Linear(in_size, 1),
        )

    def __call__(self, x):
        return F.sigmoid(self.l1(x))
