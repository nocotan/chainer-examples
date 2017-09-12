# -*- coding: utf-8 -*-
import glob
import os

import chainer
import numpy as np
import scipy
import skimage


class Pix2PixDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, crop_size, train=True,
                 use_original=False, ext='jpg'):
        self.paths = list(sorted(glob.glob(os.path.join(path, '*.' + ext))))
        self.crop_size = crop_size
        self.train = train
        self.use_original = use_original

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        image_size = (1080, 1920)
        image_path = self._paths[i]
        image = skimage.io.imread(image_path)
        image = scipy.misc.imresize(image, image_size)

        if self.train:
            h, w, c = image.shape
            crop_x = np.random.randint(h-self.crop_size)
            crop_y = np.random.randint(w-self.crop_size)
            image = image[crop_x:crop_x+self.crop_size,
                          crop_y:crop_y+self.crop_size]

            if np.random.random() < 0.5:
                image = np.fliplr(image)

            if self.use_original:
                color_image_dir, color_image_fn = os.path.split(image_path)
                orig_image_file = os.path.join(color_image_dir, 'orig',
                                               color_image_fn)
                orig_image = skimage.io.imread(orig_image_file, as_grey=False)
                orig_image = scipy.misc.imresize(orig_image, image)
                return self.get_orig_color_pair(orig_image, image)
            else:
                return self.get_grey_clor_pair(image)

    def get_random_example(self):
        return self.get_example(np.random.randint(self.__len__()))

    def get_grey_clor_pair(self, img_arr):
        color = np.asarray(img_arr, dtype=np.float32) / 127.5 - 1
        return color.transpose(2, 0, 1), color.transpose(2, 0, 1)

    def get_orig_color_pair(self, orig_arr, color_arr):
        orig = np.asarray(orig_arr, dtype=np.float32) / 127.5 - 1
        color = np.asarray(color_arr, dtype=np.float32) / 127.5 - 1
        return (orig.transpose(2, 0, 1), color.transpose(2, 0, 1))
