# -*- coding: utf-8 -*-
import os
import time
import argparse
import cv2
import skimage
import numpy as np
import cupy as cp
import chainer
from chainer import training
from chainer.training import extensions

from dataset import Pix2PixDataset


def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def test(data_dir, gen, compare, fps, use_original):
    def array_to_image(arr):
        return np.asarray((chainer.cuda.to_cpu(
            cp.transpose((cp.clip(arr, -1, 1) + 1) * 127.5, (1, 2, 0)))),
                          dtype=np.uint8)

    width = 540
    test_imgs = Pix2PixDataset(os.path.join(data_dir, 'val'), width,
                               train=False, use_original=use_original)
    data_dir_name = os.path.split(data_dir)[1]
    video_writer = cv2.VideoWriter('./test/' + data_dir_name + '.mp4',
                                   cv2.VideoWriter_fourcc(*'H264'),
                                   fps, (1920 * 2, 1080), False)
    length = len(test_imgs)
    start = time.time()
    for img_i in range(length):
        bw, real = test_imgs.get_example(img_i)
        test_img = cp.asarray([bw])
        output = gen(test_img, False).data[0]
        if compare:
            bw_img = np.concatenate([array_to_image(bw)] * 3, axis=2)
            fake_img = array_to_image(output)
            real_img = array_to_image(real)
            output_img = np.concatenate([bw_img, fake_img, real_img], axis=1)
        else:
            output_img = array_to_image(output)

        skimage.io.imsave('./test/{}.jpg'.format(img_i), output_img)
        video_writer.write(cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        print(img_i + 1, '/', length)
    end = time.time()
    duration = end - start
    print(duration, 'sec')
    print(length / duration, 'fps')


def main():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--data_dir', type=str,
                        help='directory including train/val directories')
    parser.add_argument('--train', action='store_true', dest='is_train',
                        default=True, help='train mode')
    parser.add_argument('--test', action='store_false', dest='is_train',
                        default=True, help='test mode')
    parser.add_argument('--model', type=str, help='npz file of trained model')
    parser.add_argument('--use_original', type=str, action='store_true',
                        default=False, help='use original BW images')
    parser.add_argument('--algorithm', '-a', type=str, default='dcgan',
                        help='GAN algorithm')
    parser.add_argument('--architecture', type=str, default='dcgan',
                        help='Network architecture')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--evaluation_interval', type=int, default=10000,
                        help='Interval of evaluation')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--n_dis', type=int, default=5,
                        help='number of discriminator update per generator do')
    parser.add_argument('--gamma', type=float, default=5,
                        help='hyperparameter gamma')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty')
    parser.add_argument('--adam_alpha', type=float, default=0.0002,
                        help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0,
                        help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9,
                        help='beta2 in Adam optimizer')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='output dimension of the discriminator')

    args = parser.parse_args()
    report_keys = ['loss_dis', 'loss_gen', 'inception_mean',
                   'inception_std', 'FID']

    train_dataset = Pix2PixDataset(os.path.join(args.data_dir, 'train'), 540,
                                   train=False, use_original=args.use_original)
    train_iter = chainer.iterators.SerialIterator(train_dataset,
                                                  args.batchsize)

    models = []
    opts = {}
    updater_args = {
        'iterator': {'main': train_iter},
        'device': args.gpu
    }

    if args.algorithm == 'dcgan':
        from dcgan.update import Updater
        if args.architecture == 'dcgan':
            from dcgan import Generator, Discriminator
            generator = Generator()
            discriminator = Discriminator()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print("use gpu {}".format(args.gpu))
        for m in models:
            m.to_gpu()

    opts['opt_gen'] = make_optimizer(generator, args.adam_alpha,
                                     args.adam_beta1, args.adam_beta2)
    opts['opt_dis'] = make_optimizer(discriminator, args.adam_alpha,
                                     args.adam_beta1, args.adam_beta2)

    updater_args['optimizer'] = opts
    updater_args['models'] = models

    updater = Updater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iterator'),
                               out=args.out)

    for m in models:
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'),
                       trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(
        keys=report_keys, trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys),
                   trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == '__main__':
    main()
