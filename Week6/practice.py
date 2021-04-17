import cv2
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from using_myfunc.my_gaussian import my_filtering


def get_LoG_filter(fsize, sigma=1):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2): (fsize // 2) + 1]
    LoG = (1 / (2 * np.pi * sigma ** 2)) * (((x ** 2 + y ** 2) / sigma ** 4) - (2 / sigma ** 2)) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    LoG = LoG - (LoG.sum() / fsize ** 2)
    cv2.imshow('LoG', LoG)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # return LoG
    return np.round(LoG, 3)


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    print(src)
    # src = src / 255  # 0 ~ 1
    print(src)
    LoG_filter = get_LoG_filter(fsize=9, sigma=1)

    print(LoG_filter)

    dst = my_filtering(src, LoG_filter, 'zero')
    print(dst)
    dst = dst/255
    cv2.imshow('filtering', dst)
    print(dst.max(), dst.min())

    dst = np.abs(dst)
    dst = dst - dst.min()
    dst = dst / dst.max()
    print(dst.max(), dst.min())

    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()

    src = cv2.imread('../imgs/double_threshold_test.png',cv2.IMREAD_GRAYSCALE)
    # src -= src.min()
    # src /= src.max()
    # src *= 255
    # src = src.astype(np.uint8)
    print(np.clip(np.add([0,128,255],[128,128,128]),0,255))
    cv2.imshow('src',src)
    print('start\n',src,'\nend')
    ret1, dst1 = cv2.threshold(src , 80 , 127 , cv2.THRESH_BINARY)
    ret2, dst2 = cv2.threshold(src , 200 , 255 , cv2.THRESH_BINARY)
    print(ret1,ret2)
    cv2.imshow('dst1',dst1)
    cv2.imshow('dst2',dst2)
    cv2.imshow('qweqweqwe', dst1 + dst2)
    print(cv2.add(dst1,dst2))
    cv2.waitKey()
    cv2.destroyAllWindows()

