from pathlib import Path
import os
import cv2
import torch
from options import opt
import numpy as np

CLIPPATH = "D:/mike/github/Lane_Detection/data/video/frame"
VIDSAVEPATH = "D:/mike/github/Lane_Detection/data/video/vid/video.avi"
fps = 30

frame_array = []
files = [f for f in os.listdir(CLIPPATH) if os.path.isfile(os.path.join(CLIPPATH, f))]
files.sort()

print('Read frames...')
for i in range(len(files)):
    print(files[i])
    filename=os.path.join(CLIPPATH, files[i])
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(VIDSAVEPATH,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

print('Generate video...')

for i in range(len(frame_array)):
    # writing to a image array
    print(i)
    out.write(frame_array[i])
out.release()