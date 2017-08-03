# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L

class AlexNet(chainer.Chain):
    def __init__(self, num_class, train=True):
        super(AlexNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 96, 11, stride=4)
            self.conv2=L.Convolution2D(None, 256, 5, pad=2)
            self.conv3=L.Convolution2D(None, 384, 3, pad=1)
            self.conv4=L.Convolution2D(None, 384, 3, pad=1)
            self.conv5=L.Convolution2D(None, 256, 3, pad=1)
            self.fc6=L.Linear(None, 4096)
            self.fc7=L.Linear(None, 4096)
            self.fc8=L.Linear(None, num_class)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h
