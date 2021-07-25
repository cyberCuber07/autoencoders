from src.__init__ import *
from src.params import INPUT_SHAPE
from src.model import Denoise
import os
import cv2
import tensorflow as tf
import numpy as np


"""code's working only for WEIGHTS now!!"""


def load_model(model_weights):
    model = Denoise()
    model = model.model
    model.load_weights(model_weights)
    return model


def read_img(img_path, gray=True):
    if gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)
    img = cv2.resize(img, INPUT_SHAPE[:2])
    img = img.astype('float32') / 255.
    if gray:
        img = img[..., tf.newaxis]
        # img = img[tf.newaxis, ..., tf.newaxis]
    else:
        img = img[tf.newaxis, ...]
    return img


def load_imgs(imgs_path):
    imgs = []
    for _, _, files in os.walk(imgs_path):
        for file in files:
            img = read_img(os.path.join(imgs_path, file))
            imgs.append(img)
    return np.array(imgs)
