
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


INPUT_SHAPE = (28,28, 1)
BATCH_SIZE = 32
CLASS_MODE = "categorical"
NOISE = 0.2

EPOCHS = 10
LOSS = MeanSquaredError()
LR = 10 ** -3
OPTIMIZER = Adam(lr=LR)


MODEL_DIR = "models"
FIGURE_DIR = "figures"
