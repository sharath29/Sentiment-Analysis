from constants import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image


FILE_PATH = 'fer2013.csv'
data = pd.read_csv(FILE_PATH)


def data_to_image(data):
    data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = cv2.resize(data_image,(250,250))
    return data_image


for index, row in data.iterrows():
  image = data_to_image(row['pixels'])
  cv2.imshow('Video', image)
  k = cv2.waitKey(0)
  if k == 27: break

cv2.destroyAllWindows()
