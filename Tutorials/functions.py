# -*- coding: utf-8 -*-
import numpy as np
from chainer import Variable
import chainer.functions as F

x = Variable(np.array([-1]).astype(np.float32))
print(F.sin(x).data)
print(F.sigmoid(x).data)

x = Variable(np.array([-0.5]).astype(np.float32))
z = F.cos(x)
print(z.data)

z.backward()

print(x.grad)
print(((-1)*F.sin(x).data))

x = Variable(np.array([-1, 0, 1]).astype(np.float32))
z = F.sin(x)
z.grad = np.ones(3, dtype=np.float32)
z.backward()
print(x.grad)
