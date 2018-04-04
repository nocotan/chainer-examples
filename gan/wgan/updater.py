# -*- coding: utf-8 -*-
from chainer import Variable
from chainer.training import StandardUpdater


class Updater(StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.critic = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')
        critic_optimiser = self.get_optimizer('critic')

        xp = self.gen.xp

        for _ in range(5):
            self.critic.clamp()
            x_real = self.get_iterator('main').next()
            batchsize = len(x_real)

            y_real = self.critic(x_real)
            y_real.grad = xp.ones_like(y_real.data)

            self.critic.cleargrads()
            y_real.backward()
            critic_optimiser.update()

            z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
            x_fake = self.gen(z)
            y_fake = self.critic(x_fake)
            y_fake.grad = -1 * xp.ones_like(y_fake.data)

            self.critic.cleargrads()
            y_fake.backward()
            critic_optimiser.update()

        z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        x_fake = self.gen(z)
        y_fake = self.critic(x_fake)
        y_fake.grad = xp.ones_like(y_fake.data)

        self.gen.cleargrads()
        y_fake.backward()
        gen_optimizer.update()
