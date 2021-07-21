from src.__init__ import *
from src.model import Denoise
from src.train.utils import get_training_batch, get_dirs_num
from src.params import *
from src.utils import plot_history, get_name
import argparse
import os
from tqdm import tqdm


def main():
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    for epoch in range(EPOCHS):
        names = ['train', 'valid']
        num_iter = get_dirs_num(args.path, "train")
        for iter in tqdm(range(num_iter)):
            num_imgs, x_train_noisy, x_train = get_training_batch(iter, args.path)
            history = autoencoder.fit(x_train_noisy, x_train,
                                      epochs=1,
                                      shuffle=True,
                                      batch_size=SMALL_BATCH_SIZE,
                                      validation_split=VALIDATION_SPLIT,
                                      verbose=0)

        plot_history(history)

        if epoch % 5 == 0:
            autoencoder.save_weights(get_name(WEIGHTS_DIR, ".h5"))


if __name__ == "__main__":
    args = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    args.add_argument("--path", type=str, default="")
    args.add_argument("--images", type=str, default="images")
    args.add_argument("--weights", type=str, default="weights/4.h5")
    args = args.parse_args()

    autoencoder = Denoise()
    autoencoder = autoencoder.model
    autoencoder.summary()
    main()
