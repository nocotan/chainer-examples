# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L

class LeNet(chainer.Chain):
    def __init__(self, num_class, train=True):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 6, 5, stride=1)
            self.conv2=L.Convolution2D(None, 16, 5, stride=1)
            self.fc3=L.Linear(None, 120)
            self.fc4=L.Linear(None, 64)
            self.fc5=L.Linear(None, num_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.sigmoid(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.sigmoid(self.conv2(h))), 2, stride=2)
        h = F.sigmoid(self.fc3(h))
        h = F.sigmoid(self.fc4(h))
        h = self.fc5(h)

        return h

