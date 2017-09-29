# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self, n_hidden=128, bottom_width=4, ch=512, wscale=0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width*bottom_width*ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch//2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch//2, ch//4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch//4, ch//8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch//8, 3, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(bottom_width*bottom_width*ch)
            self.bn1 = L.BatchNormalization(ch//2)
            self.bn2 = L.BatchNormalization(ch//4)
            self.bn3 = L.BatchNormalization(ch//8)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))),
                      (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = F.tanh(self.dc4(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self, ch=512, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        self.ch = ch
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.l4 = L.Linear(4*4*ch, 128, initialW=w)
            self.l5 = L.Linear(128, 4*4*ch, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 1, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc1 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc0 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = x
        h = F.leaky_relu(self.c0(h))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c3(h))
        h = F.leaky_relu(self.l4(h))
        h = F.reshape(F.leaky_relu(self.l5(h)),
                      (x.data.shape[0], self.ch, 4, 4))
        h = F.leaky_relu(self.dc3(h))
        h = F.leaky_relu(self.dc2(h))
        h = F.leaky_relu(self.dc1(h))
        h = F.tanh(self.dc0(h))
        return F.mean_absolute_error(h, x)
