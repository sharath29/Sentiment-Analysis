from __future__ import division, absolute_import
import numpy as np
from dataset_loader import DatasetLoader
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.models import load_model
from constants import *
from os.path import isfile, join
import random
import sys

# model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
# validation_split=0.3, shuffle=True, verbose=1)


class EmotionRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()

  def build_network(self):
    print('[+] Building CNN')
    
    self.model = Sequential()
    self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu',
                            input_shape=(48, 48, 1)))
    self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    self.model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    self.model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dense(7, activation='softmax'))
    # optimizer:
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print 'Training....'
    
    self.load_model()

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('[+] Dataset found and loaded')

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('[+] Training network')
    self.model.fit(
      self.dataset.images, self.dataset.labels,
      validation_data = (self.dataset.images_test, self.dataset._labels_test),
      epochs = 3,
      batch_size = 50,
      shuffle = True,
      verbose = 1
    )

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

  def load_model(self):
    if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME)):
      self.model = load_model(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
      print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)


def show_usage():
  # I din't want to have more dependecies
  print('[!] Usage: python emotion_recognition.py')
  print('\t emotion_recognition.py train \t Trains and saves model with saved dataset')
  print('\t emotion_recognition.py poc')

if __name__ == "__main__":
  if len(sys.argv) <= 1:
    show_usage()
    exit()

  network = EmotionRecognition()
  if sys.argv[1] == 'train':
    network.start_training()
    network.save_model()
  elif sys.argv[1] == 'run':
    import run
  else:
    show_usage()
