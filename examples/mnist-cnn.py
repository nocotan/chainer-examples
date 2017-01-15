# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


class CNN(chainer.Chain):
    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),
            conv2=L.Convolution2D(32, 64, 5),
            l1=L.Linear(1024, 10),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        return self.l1(h)

def main():
    model = L.Classifier(CNN())

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=100)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (5, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy','validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
