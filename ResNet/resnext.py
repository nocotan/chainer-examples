# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Cardinality(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride):
        super(Cardinality, self).__init__()
        w = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_size, ch, 1, stride, 0,
                                         initialW=w, nobias=True)
            self.conv2 = L.Convolution2D(ch, ch, 3, 1, 1,
                                         initialW=w, nobias=True)
            self.conv3 = L.Convolution2D(ch, out_size, 1, 1, 0,
                                         initialW=w, nobias=True)

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h = F.elu(self.conv2(h))
        return self.conv3(h)


class Block(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride, card=32):
        super(Block, self).__init__()
        links = [(
            'c{}'.format(i+1),
            Cardinality(in_size, ch, out_size, stride)) for i in range(card)]
        links += [(
            'x_bypass',
            L.Convolution2D(in_size, out_size, 1, stride, 0, nobias=True))]

        for l in links:
            self.add_link(*l)

        self.forward = links

    def __call__(self, x):
        h = None
        for name, _ in self.forward:
            f = getattr(self, name)
            h_t = f(x)
            if h is None:
                h = h_t
            else:
                h += h_t

        return F.elu(h)


class LaminationBlock(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=1):
        super(LaminationBlock, self).__init__()
        links = [('lb0', Block(in_size, ch, out_size, stride))]
        links += [('lb{}'.format(i+1),
                   Block(out_size, ch, out_size, 1)) for i in range(1, layer)]
        for l in links:
            self.add_link(*l)

        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x)

        return x


class ResNext(chainer.Chain):
    def __init__(self, num_class):
        self.train = True
        self.num_class = num_class
        w = initializers.HeNormal()
        super(ResNext, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 128, 7, 2, 3,
                                         initialW=w, nobias=True)
            self.res2 = LaminationBlock(3, 128, 4, 256, stride=2)
            self.res3 = LaminationBlock(4, 256, 4, 512, stride=2)
            self.res4 = LaminationBlock(6, 512, 256, 1024)
            self.res5 = LaminationBlock(3, 1024, 512, 2048)
            self.conv2 = L.Convolution2D(None, 4096, 1, pad=0)
            self.conv3 = L.Convolution2D(None, num_class, 1, pad=0)

    def __call__(self, x):
        x.volatile = not self.train

        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)

        h = F.elu(self.conv2(h))
        h = F.dropout(h, ratio=0.5, train=self.train)
        h = self.conv3(h)
        h = F.reshape(h, (-1, self.num_class))

        return h
