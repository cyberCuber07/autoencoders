from src.__init__ import *
import os
import numpy as np
from tensorflow.keras.datasets.mnist import load_data as load_mnist
from keras_preprocessing.image import ImageDataGenerator
from src.params import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2


def load_data(path=None, valid=True):
    if path is None:
        # (x_train, _), (x_test, _) = load_mnist()
        #
        # for idx, _ in enumerate(x_train):
        #     cv2.imwrite(os.path.join("train", str(idx) + ".jpg"), x_train[idx])
        # for idx, _ in enumerate(x_test):
        #     cv2.imwrite(os.path.join("test", str(idx) + ".jpg"), x_test[idx])

        x_train, x_test = [], []
        for idx in range(60000):
            img = cv2.imread(os.path.join("train", str(idx) + ".jpg"), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, INPUT_SHAPE[:2])
            x_train.append(img)
        for idx in range(10000):
            img = cv2.imread(os.path.join("test", str(idx) + ".jpg"), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, INPUT_SHAPE[:2])
            x_test.append(img)
        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        x_train_noisy = x_train + NOISE * tf.random.normal(x_train.shape)
        x_test_noisy = x_test + NOISE * tf.random.normal(x_test.shape)

        x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
        x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

        x_train_noisy = np.array(x_train_noisy)
        x_test_noisy = np.array(x_test_noisy)

        # sample
        # ------------
        num = 10
        for img in x_train[:num]:
            img = img + NOISE * tf.random.normal(img.shape)
            img = tf.clip_by_value(img, clip_value_min=0., clip_value_max=1.)
            img = img * 255
            img = np.array(img)
            cv2.imwrite(get_name(IMAGES_DIR, ".jpg"), img)

        return (x_train_noisy, x_train), (x_test_noisy, x_test)

    else:
        validation_generator = None
        if valid:
            valid_path = os.path.join(path, "valid")
            path = os.path.join(path, "train")

        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest')

        data_generator = datagen.flow_from_directory(
            path,
            target_size=INPUT_SHAPE,
            batch_size=BATCH_SIZE,
            class_mode=CLASS_MODE,
            color_mode=COLOR_MODE[CHANNELS],
            shuffle=True)

        if valid:
            validation_datagen = ImageDataGenerator(rescale=1. / 255)
            validation_generator = validation_datagen.flow_from_directory(
                valid_path,
                target_size=INPUT_SHAPE,
                batch_size=BATCH_SIZE,
                class_mode=CLASS_MODE,
                color_mode=COLOR_MODE[CHANNELS],
                shuffle=False)
        return data_generator, validation_generator


def get_name(save_dir, file_type):
    max_idx = 0
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    for _, _, files in os.walk(save_dir):
        for file in files:
            max_idx = max(max_idx, int(file[:-len(file_type)]))
    return os.path.join(save_dir, (str(max_idx + 1) + file_type))


def plot_history(history=None):
    def plot_one(type):
        plt.plot(history.history[type])
        plt.plot(history.history['val_' + type])
        plt.title('model ' + type)
        plt.ylabel(type)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(get_name(FIGURE_DIR, ".jpg"))
        plt.close()
    # define function pointer
    if history is None:
        return
    print(history.history.keys())
    plot_one("loss")
