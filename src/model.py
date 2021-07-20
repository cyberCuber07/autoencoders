from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose,
                                     BatchNormalization, ReLU, MaxPool2D)
from tensorflow.keras.models import Model, Sequential
from src.params import *
from icecream import ic


class __denoise__:
    def __init__(self):
        self.intput_shape = INPUT_SHAPE
        self.conv_num = CONV_NUM
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

    def create_model(self):
        input = Input(self.intput_shape)
        f_max = F_MAX
        filters = [512, 256, 128, 64, 8, 8, 64, 128, 256, 512]
        self.conv_num = len(filters) // 2
        x = Conv2D(f_max, kernel_size=1, strides=1)(input)
        for idx in range(self.conv_num):
            f_max = f_max // 2
            x = self.conv_block(False, x, filters[idx], 1, 2)
        for idx in range(self.conv_num):
            f_max *= 2
            x = self.conv_block(True, x, filters[idx + self.conv_num], 1, 2)
        output = Conv2DTranspose(1, 1, 1)(x)
        return Model(input, output)
