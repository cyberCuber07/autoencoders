from src.model import Denoise
from src.params import *
from src.utils import *
from src.test import test
from icecream import ic
import numpy as np
import argparse


def main():
    # autoencoder.summary()
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    history = autoencoder.fit(train_data[0], train_data[1],
                              epochs=EPOCHS,
                              shuffle=True,
                              validation_data=(valid_data[0], valid_data[1]))

    plot_history(history)
    autoencoder.encoder.save(get_name(MODEL_DIR, ".h5"))
    autoencoder.decoder.save(get_name(MODEL_DIR, ".h5"))


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--train", action="store_true", default=False)
    args.add_argument("--test", action="store_true", default=False)
    args.add_argument("--images", type=str, default="images")
    args.add_argument("--model", type=str, default="models/1.h5")
    args = args.parse_args()

    if args.train:
        train_data, valid_data = load_data()
        autoencoder = Denoise()
        main()

    if args.test:
        test(args.model, args.images)