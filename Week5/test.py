import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_gaussian import my_get_Gaussian2D_mask
from my_filtering import my_filtering

fsize = 256
sigma = 30
print(np.math.e)
# y, x = np.mgrid[-(fsize // 2):1 + (fsize // 2), -(fsize // 2):1 + (fsize // 2)]
#
# DoG_x = -(x / (sigma ** 2)) * (np.math.e ** (- (x * x + y * y) / (2 * sigma * sigma)))
# DoG_x = ((DoG_x - np.min(DoG_x)) / np.max(DoG_x - np.min(DoG_x))*255).astype(np.uint8)
# DoG_y = -(y / (sigma ** 2)) * (np.math.e ** (- (x * x + y * y) / (2 * sigma * sigma)))
# DoG_y = ((DoG_y - np.min(DoG_y)) / np.max(DoG_y - np.min(DoG_y))*255).astype(np.uint8)
#
# print('----------')
# print(DoG_x)
# print('----------')
#
# print(DoG_y)
# cv2.imshow('DoG-x', DoG_x)
# cv2.imshow('Dog-y', DoG_y)
# cv2.waitKey()
# cv2.destroyAllWindows()