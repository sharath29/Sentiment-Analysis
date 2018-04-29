import cv2
import os

org, font, scale, color, thickness, linetype = (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (234,12,123), 2, cv2.LINE_AA
cap = cv2.VideoCapture('http://192.168.42.129:8080/video')
path = os.getcwd() + '/dataset/'
data_size = 300

gesture = ''
while True:
  _, img = cap.read()
  cv2.putText(img, 'enter emotion name', org, font, scale, color, thickness, linetype)
  cv2.putText(img, 'press esc when finished', (50,100), font, scale, color, thickness, linetype)
  cv2.putText(img, gesture, (50,300), font, 3, (0,0,255), 5, linetype)
  cv2.line(img, (330,240), (310,240), (234,123,234), 3)
  cv2.line(img, (320,250), (320,230), (234,123,234), 3)
  cv2.imshow('img1', img)
  k = cv2.waitKey(5)
  if k>=97 and k <= 122: gesture += chr(k)
  if k == 27: break
  if k == 13:
    current = 0
    dirname = gesture.upper()
    os.mkdir(path + dirname)
    while current < data_size:
      _, img = cap.read()
      cv2.imwrite(os.path.join(path , dirname , str(current) + '.jpg'), img)
      cv2.putText(img, 'face is '+ gesture.upper() +': ' + str(current), (50,100), font, scale, color, thickness, linetype)
      cv2.imshow('data collection', img)
      k = cv2.waitKey(5)
      if k == 27: break
      current += 1
      print current
    gesture = ''
      
cv2.destroyAllWindows()














