import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from using_myfunc.my_gaussian import my_get_Gaussian2D_mask
from using_myfunc.my_gaussian import my_filtering
print(30 * np.sqrt(np.math.e))
src = cv2.imread('../imgs/sobel_test.png', cv2.IMREAD_GRAYSCALE)
print(src)
src_f = src.astype(np.float32)
fsize = 256
sigma = 30
sb = np.dot([[-1,0,1]],my_get_Gaussian2D_mask(3,sigma))
print(sb)
der = np.array([[-1,0,1]])
blur = np.array([[1],[2],[1]])
sobel_x = np.dot(blur,der)
sobel_y = np.dot(der.T,blur.T)
print(np.sum(sobel_y),np.sum(sobel_x))


dst_x = my_filtering(src,sobel_x,'zero')
dst_y = my_filtering(src,sobel_y,'zero')
dst_x = np.abs(dst_x)
dst_y = np.abs(dst_y)
dst = dst_x+dst_y
xx = np.clip(dst_x,0,255).astype(np.uint8)
yy = np.clip(dst_y,0,255).astype(np.uint8)
cv2.imshow('xx',xx)
cv2.imshow('yy',yy)
cv2.imshow('dst',dst/255)
cv2.imshow('dst_x',dst_x/255)
cv2.imshow('dst_y',dst_y/255)

cv2.waitKey()
cv2.destroyAllWindows()