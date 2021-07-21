from src.model import __denoise__
from src.params import *
from src.utils import *
from src.test import test
from icecream import ic
import numpy as np
import argparse


def main():
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    if args.path is None:
        history = autoencoder.fit(train_data[0], train_data[1],
                                  epochs=EPOCHS,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  validation_data=(valid_data[0], valid_data[1]))
    else:
        history = autoencoder.fit(train_data,
                                  epochs=EPOCHS,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  validation_data=valid_data)


    plot_history(history)
    autoencoder.save_weights(get_name(WEIGHTS_DIR, ".h5"))


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--train", action="store_true", default=False)
    args.add_argument("--test", action="store_true", default=False)
    args.add_argument("--path", type=str, default=None)
    args.add_argument("--images", type=str, default="images")
    args.add_argument("--weights", type=str, default="weights/4.h5")
    args = args.parse_args()

    if args.train:
        train_data, valid_data = load_data(args.path)
        autoencoder = __denoise__()
        autoencoder = autoencoder.model
        autoencoder.summary()
        main()

    if args.test:
        test(args.weights, args.images)