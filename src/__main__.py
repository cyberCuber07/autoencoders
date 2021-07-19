from src.model import Denoise
from src.params import *
from src.utils import *
from icecream import ic
import numpy as np


def main():
    # autoencoder.summary()
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS)

    history = autoencoder.fit(train_data[0], train_data[1],
                              epochs=EPOCHS,
                              shuffle=True,
                              validation_data=(valid_data[0], valid_data[1]))

    plot_history(history)
    autoencoder.save_model(get_name(MODEL_DIR, ".jpg"))


if __name__ == "__main__":
    train_data, valid_data = load_data()

    autoencoder = Denoise()

    main()