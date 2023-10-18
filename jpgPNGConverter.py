import cv2
import os

path = '/home/thws_robotik/Documents/Leyh/6dpose/datasets/own/masks/'
imgs = os.listdir(path)
imgs.sort()
for entry in imgs:
  print(f"reading {entry}")
  if not entry.endswith('.jpg'):
    continue
  frame = cv2.imread(path + "/" + entry)
  frame = cv2.resize(frame, (2560, 1440)) 
  destfile = entry.replace(".jpg",".png")
  print(f"saving  {destfile}")
  cv2.imwrite(path + "/" + destfile, frame)