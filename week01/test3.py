import cv2
import numpy as np

src = np.zeros((300,300,3),dtype=np.uint8)
src[0:100,0:100,0] = 255
src[0:100,0:300,1] = 255
src[0:100,0:300,2] = 100

cv2.imshow('src',src)
cv2.waitKey()