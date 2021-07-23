from src.__init__ import *
from keras_preprocessing.image import ImageDataGenerator
from src.params import *
from src.utils import get_name
import os
import cv2
import numpy as np
import tensorflow as tf
import time
from contextlib import redirect_stdout


def get_dirs_num(path, name):
    type_path = os.path.join(path, name)
    for _, dirs, _ in os.walk(type_path):
        return len(dirs)


def get_training_batch(iter, path, sample=False):
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


    tmp = tf.__version__
    if tmp[0] == '1':
        x_train_noisy = tf.Session().run(x_train_noisy)
    elif tmp[1] == '2':
        x_train_noisy = x_train_noisy.numpy()
    ic(x_train_noisy.shape)

    if sample:
        num = 10
        for idx in range(num):
            img_path = get_name(IMAGES_DIR, ".jpg")
            img = x_train_noisy[idx] * 255.
            cv2.imwrite(img_path, img)

    return num_imgs, x_train_noisy, x_train


def summary2file(model, name, _type):
    name = name.split("/")[-1][:-len(_type)]
    save_dir = os.path.join(SUMMARY_DIR, name)
    imgs_dir = os.path.join(save_dir, "imgs")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(imgs_dir):
        os.mkdir(imgs_dir)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()    # main()
    return save_dir


def log(epoch, history, start_time):
    def conv_time():
        full_time = int(time.time() - start_time)
        if full_time <= 3600:
            return time.strftime('%M:%S', time.gmtime(full_time))
        else:
            return time.strftime('%H:%M:%S', time.gmtime(full_time))

    total_time = conv_time()
    loss = np.mean(np.array([float(one.history['loss'][0]) for one in history]))
    val_loss = np.mean(np.array([float(one.history['val_loss'][0]) for one in history]))

    info = "Epoch: {}; Time it took: {:.2f}; loss={:.4f}; val_loss={:.4f}".\
        format(epoch, total_time, loss, val_loss)

    print("------------------------------------------------------------------------", end='')
    print("----------------------------------------")
    print(info)
    print("------------------------------------------------------------------------", end='')
    print("----------------------------------------")
