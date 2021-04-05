
import numpy as np
import cv2
import time

# library add
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from padding import my_padding



def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################

    y, x = np.mgrid[-(msize//2):1+(msize//2),-(msize//2):1+(msize//2)]
    # y = [[-1,-1,-1],
    #      [ 0, 0, 0],
    #      [ 1, 1, 1]]
    # x = [[-1, 0, 1],
    #      [-1, 0, 1],
    #      [-1, 0, 1]]
    f = x*x+y*y
    # f = [[2,1,2],
    #      [1,0,1],
    #      [2,1,2]]

    # 2차 gaussian mask 생성
    gaus2D = ((np.math.e)**( -(f)/2))/2*np.math.pi


    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D


def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################

    y, x = np.mgrid[0:1, -(msize // 2):1 + (msize // 2)]
    f = x*x+y*y
    gaus1D = (np.math.e)**(-(f/(2*sigma*sigma)))/np.math.sqrt(2*np.math.pi)
    # mask의 총 합 = 1
    gaus1D /= np.sum(gaus1D)
    return gaus1D


def my_filtering(src, mask, pad_type='zero'): # 이미지, 마스크, 패딩타입
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape #(1,5) (5,1)
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    # print('<mask>')
    # print(mask)

    # 시간을 측정할 때 만 이 코드를 사용하고 시간측정 안하고 filtering을 할 때에는
    # 4중 for문으로 할 경우 시간이 많이 걸리기 때문에 2중 for문으로 사용하기.
    dst = np.zeros((h, w))
    #dst = np.sum(src * mask)



    for y in range(h):
        for x in range(w):
            dst[y,x] = np.sum(np.multiply(mask,pad_img[y:y+m_h,x:x+m_w]))
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    mask_size = 111
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma=1)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma=1)
    print('mask size : ', mask_size)
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus1D = my_filtering(src, gaus1D.T)
    dst_gaus1D = my_filtering(dst_gaus1D, gaus1D)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end - start)

    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus2D = my_filtering(src, gaus2D)
    end = time.perf_counter()  # 시간 측정 끝
    print('2D time : ', end - start)
    # 마스크의 크기를 99로 했을 때 2차원의 시간측정 값이 1차원의 2배
    dst_gaus1D = np.clip(dst_gaus1D + 0.5, 0, 255)
    dst_gaus1D = dst_gaus1D.astype(np.uint8)
    dst_gaus2D = np.clip(dst_gaus2D + 0.5, 0, 255)
    dst_gaus2D = dst_gaus2D.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('1D gaussian img', dst_gaus1D)
    cv2.imshow('2D gaussian img', dst_gaus2D)
    cv2.waitKey()
    cv2.destroyAllWindows()