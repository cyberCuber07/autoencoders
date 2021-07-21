from src.__init__ import *
from keras_preprocessing.image import ImageDataGenerator
from src.params import *
import os
import cv2
import numpy as np
import tensorflow as tf


def get_dirs_num(path, name):
    type_path = os.path.join(path, name)
    for _, dirs, _ in os.walk(type_path):
        return len(dirs)


def get_training_batch(iter, path):
    """
    method to get specified batch of images
    directories are addressed using 'iter' variable
    """
    def get_imgs_num(_path):
        for _, _, files in os.walk(_path):
            return len(files)

    def read_img(img_path):
        if CHANNELS == 3:
            img = cv2.imread(img_path)
        elif CHANNELS == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, INPUT_SHAPE)

    def read_imgs():
        x_train = []
        for _, _, files in os.walk(path):
            for _file in files:
                img_path = os.path.join(path, _file)
                img = read_img(img_path)
                x_train.append(img)

        x_train = np.array(x_train)
        if CHANNELS == 1:
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

        x_train = x_train.astype('float32') / 255.
        return x_train

    path = os.path.join(path, "train", str(iter))
    num_imgs = get_imgs_num(path)
    x_train = read_imgs()

    x_train_noisy = x_train + NOISE * tf.random.normal(x_train.shape)
    x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)

    return num_imgs, x_train_noisy, x_train
