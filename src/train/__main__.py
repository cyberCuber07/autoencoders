from src.__init__ import *
from src.model import Denoise
from src.train.utils import get_training_batch, get_imgs_num
from src.params import *
from src.utils import plot_history, get_name
import argparse
import os


def main():
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    for epoch in range(EPOCHS):
        names = ['train', 'valid']
        num_imgs = get_imgs_num(os.path.join(args.path, "train"), names)
        num_iter = num_imgs // BATCH_SIZE
        for iter in range(num_iter):
            train_batch, valid_batch = get_training_batch(iter, args.path)
            history = autoencoder.fit(train_batch,
                                      epochs=1,
                                      shuffle=True,
                                      batch_size=SMALL_BATCH_SIZE,
                                      validation_data=valid_batch)

        plot_history(history)

        if epoch % 5 == 0:
            autoencoder.save_weights(get_name(WEIGHTS_DIR, ".h5"))


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--path", type=str, default=None)
    args.add_argument("--images", type=str, default="images")
    args.add_argument("--weights", type=str, default="weights/4.h5")
    args = args.parse_args()

    autoencoder = Denoise()
    autoencoder = autoencoder.model
    # autoencoder.summary()
    main()
