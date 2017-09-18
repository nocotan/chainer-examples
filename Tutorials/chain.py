# -*- coding: utf-8 -*-
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Variable


class ToyChain(chainer.Chain):
    def __init__(self):
        super(ToyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2),
        )

    def __call__(self, x, t):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        y = F.mean_squared_error(h, t)
        return y


x = Variable(np.arange(12).astype(np.float32).reshape(3, 4))
y = Variable(np.array([[0, 1], [1, 0], [1, 0]]).astype(np.float32))
print(x.data)

model = ToyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

model.cleargrads()
loss = model(x, y)
print(loss)
loss.backward()
optimizer.update()
