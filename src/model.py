from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from src.params import INPUT_SHAPE


class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = Sequential([
      Input(shape=INPUT_SHAPE),
      Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = Sequential([
      Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
