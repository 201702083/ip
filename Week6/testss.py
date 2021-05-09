import cv2
import numpy as np

import os, sys
from my_library.DoG import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.my_gaussian import my_filtering

if __name__ == '__main__':
    # main()
    src = cv2.imread('../imgs/Lena.png',cv2.IMREAD_GRAYSCALE)
    h,w = src.shape
    Ix,Iy = get_DoG_filter(3,1)
    print(np.tan(0.785398))
    # Ix = my_filtering(src,Ix,'zero')
    # Iy = my_filtering(src,Iy,'zero')
    # cv2.imshow('Ix',np.clip(Ix,0,255))
    # cv2.imshow
    # magnitude = np.sqrt(Ix**2 + Iy**2)
    # e = 1E-6
    # angle = np.arctan(Iy/(Ix+e)) # rad -> 엣지와 수직인 방향
    # print(angle)
    # verti = angle - np.pi/2
    # print(verti)
    # # degree
    # y,x = np.mgrid[-(h//2):1+(h//2),-(w//2):1+(w//2)]
    # #linear test
    # # for row in range(h):
    # #     for col in range(w):  #  ( col  , row ) 일 때