from src.__init__ import *
from keras_preprocessing.image import ImageDataGenerator
from src.params import *
import os


def get_imgs_num(path, names):
    cnt = 0
    for name in names:
        type_path = os.path.join(path, name)
        for _, _, files in os.walk(type_path):
            cnt += len(files)
    return cnt


def get_training_batch(iter, path, valid=True):
    """
    method to get specified batch of images
    directories are addressed using 'iter' variable
    """
    validation_generator = None
    if valid:
        valid_path = os.path.join(path, "valid", str(iter))
        path = os.path.join(path, "train", str(iter))

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
