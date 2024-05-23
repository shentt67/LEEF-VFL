import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

class Mask(object):
    def __init__(self, p):
        # mask rate
        self.p = p

    def __call__(self, img):
        img_arr = np.array(img)
        mask = np.where(img_arr > 0, 1, 0)
        indexes = np.argwhere(mask == 1)
        indexes = indexes[np.random.choice(indexes.shape[0], size=int(indexes.shape[0] * self.p)), :]
        indexes = indexes.T
        mask[indexes[0], indexes[1]] = 0

        img_result = Image.fromarray(img_arr)
        return img_result


class Noise(object):
    def __init__(self, miu, sigma):
        # mask rate
        self.miu = miu
        self.sigma = sigma

    def __call__(self, img):
        img_arr = np.array(img)
        img_arr = img_arr + self.miu + np.random.randn(28, 28) * self.sigma
        img_result = Image.fromarray(img_arr)
        return img_result
