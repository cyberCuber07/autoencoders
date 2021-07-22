from src.__init__ import *
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (256, 256)
CHANNELS = 1
CLASS_MODE = "categorical"
NOISE = 0.2


# FILTERS = [64, 64, 64, 64, 64, 32, 32, 32, 32, 16, 16]
FILTERS = [128, 128, 64, 64, 64, 32, 32, 32]
LOSS = MeanSquaredError()
LR = 10 ** -3
OPTIMIZER = Adam(lr=LR)


VALIDATION_SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 32
SMALL_BATCH_SIZE = 2


WEIGHTS_DIR = "weights"
MODEL_DIR = "models"
FIGURE_DIR = "figures"
IMAGES_DIR = "images"
PREDICT_DIR = "predicted_images"
SUMMARY_DIR = "model_info"
