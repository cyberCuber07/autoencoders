
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (32, 32, 1)
# INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 16
CLASS_MODE = "categorical"
NOISE = 0.2
CONV_NUM = 3
F_MAX = 256

EPOCHS = 20
LOSS = CategoricalCrossentropy()
LR = 10 ** -3
OPTIMIZER = Adam(lr=LR)


WEIGHTS_DIR = "weights"
MODEL_DIR = "models"
FIGURE_DIR = "figures"
IMAGES_DIR = "images"
PREDICT_DIR = "predicted_images"
