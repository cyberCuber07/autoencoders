import argparse
import os
import shutil
from src.params import BATCH_SIZE

"""
    WARNING:
        number of batch images in valid is the same as in train
        MODIFY THIS
"""


"""
util module to convert images to an accessible
format for image generator in 'train' module
--------------------------------------------
images' directory is expected in format:
    ./
        ./path
            ./train
                ./1.jpg
                ./2.jpg
                ./...
            ./valid
                ./...
            ./test
                ./...
resulting directory tree:
    ./path/train
        ./1.jpg
        ./2.jpg
        ./...
    ./path/valid
        ./...
    ./path/test
        ./...
"""


NAMES = ['train', 'test']
TEMP_DIR = './.tmp_data'


def log(words):
    print(words)


def get_imgs_number(path):
    return len(os.listdir(path))


def get_new_path(path, dir_iter):
    iter_path = os.path.join(path, str(dir_iter))
    if not os.path.isdir(iter_path):
        os.mkdir(iter_path)
    return iter_path


def move_file(path, iter_path, _file):
    old_path = os.path.join(path, _file)
    new_path = os.path.join(iter_path, _file)
    try:
        shutil.move(old_path, new_path)
    except:
        pass


def convert(path):
    # log("Creating temporary directory: {}".format(TEMP_DIR))
    # os.mkdir(TEMP_DIR)
    """
    to make it work, add:
        getting number of files in a given directory ('path')
            -> number of directories:                                 num_dirs
            -> number of elements in one directory:                   num_els
            -> number of all elements in provided directory ('path'): num_imgs
            :: num_dirs = num_imgs // num_els
        create directory when counter exceeds max number
    ------------------------------------------------------
    ** for fit read only one dir at a time,
    that is for iter get str(iter)
    """
    num_imgs = get_imgs_number(path)
    num_els = BATCH_SIZE
    num_dirs = num_imgs // num_els # redundant ??!!?!?
    iter, dir_iter = 0, 0
    for _, _, files in os.walk(path):
        for _file in files:
            iter_path = get_new_path(path, dir_iter)
            move_file(path, iter_path, _file)

            iter += 1
            if iter % BATCH_SIZE == 0:
                dir_iter += 1


def main():
    imgs_path = args.imgs_path
    for name in NAMES:
        type_path = os.path.join(imgs_path, name)
        convert(type_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--imgs_path", type=str)
    args = args.parse_args()

    main()