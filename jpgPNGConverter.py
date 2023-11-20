import cv2
import os

DO_CONV = True

targetRes = [1280,720]


path = '/home/grass/Documents/Leyh/datasets/bookRealtime/depth'
outPath = '/home/grass/Documents/Leyh/datasets/bookRealtime720p/depth'
imgs = os.listdir(path)
imgs.sort()

os.makedirs(outPath, exist_ok= True)
for entry in imgs:
  print(f"reading {entry}")
  #if not entry.endswith('.jpg'):
  #  continue
  frame = cv2.imread(path + "/" + entry, cv2.IMREAD_UNCHANGED)
  destfile = entry.replace(".jpg",".png")
  print(f"saving  {destfile}")
  if DO_CONV:
    frame = cv2.resize(frame, targetRes)
  cv2.imwrite(outPath + "/" + destfile, frame)