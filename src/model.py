from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose,
                                     BatchNormalization, ReLU, MaxPool2D)
from tensorflow.keras.models import Model, Sequential
from src.params import *
from icecream import ic
from copy import deepcopy
import numpy as np


class __denoise__:
    def __init__(self):
        self.intput_shape = INPUT_SHAPE
        self.model = self.create_model()

    def conv_block(self, up, x, f, k, s, p=2):
        if up:
            x = Conv2DTranspose(filters=f, kernel_size=k, strides=s)(x)
        else:
            x = Conv2D(filters=f, kernel_size=k, strides=s)(x)
        x = ReLU()(x)
        # x = MaxPool2D(pool_size=(p, p))(x)
        x = BatchNormalization()(x)
        return x

    def get_filters(self):
        filters = deepcopy(FILTERS)
        f1 = deepcopy(filters)
        filters.reverse()
        filters = np.array([f1, filters]).reshape(2 * len(f1),)
        self.conv_num = len(filters) // 2
        return filters

    def create_model(self):
        input = Input(self.intput_shape)
        filters = self.get_filters()
        x = Conv2D(filters[0], kernel_size=1, strides=1)(input)
        for idx in range(self.conv_num):
            x = self.conv_block(False, x, filters[idx], 3, 1)
        for idx in range(self.conv_num):
            x = self.conv_block(True, x, filters[idx + self.conv_num], 3, 1)
        output = Conv2DTranspose(1, 1, 1)(x)
        return Model(input, output)
