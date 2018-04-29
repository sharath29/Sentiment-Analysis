import cv2
import numpy as np
import os
from constants import *


PATH = os.getcwd()
data_path = PATH + '/dataset'
data_dir_list = sorted(os.listdir(data_path))

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def func(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  if not len(faces) > 0:
    print "No face found"
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  return image


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print img_list
    current = 0
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    os.mkdir(PATH + '/prepro/' + dataset)
    for img in img_list:
        img_path = data_path + '/'+ dataset + '/'+ img         
        img = cv2.imread(img_path)
        img = func(cv2.imread(img_path))
        img = img*255
        cv2.imwrite(os.path.join(PATH + '/prepro' , dataset , str(current) + '.jpg'), img)
        current += 1
