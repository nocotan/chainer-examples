# -*- coding: utf-8 -*-
import numpy as np
import chainer.links as L
from chainer import Variable

h = L.Linear(3, 4)
print(h.W.data)
print(h.b.data)

x = Variable(np.array(range(6)).astype(np.float32).reshape(2, 3))
print(x.data)

y = h(x)
print(y.data)

w = h.W.data
x0 = x.data
print(x0.dot(w.T) + h.b.data)
