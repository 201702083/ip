import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from using_myfunc.my_gaussian import my_filtering
from using_myfunc.my_gaussian import my_get_Gaussian2D_mask
def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    b, a = np.mgrid[-(fsize//2):1+(fsize//2),-(fsize//2):1+(fsize//2)]
    gaus = my_get_Gaussian2D_mask(fsize,sigma)
    gaus = (( gaus - np.min(gaus))/ np.max(gaus - np.min(gaus)) * 255 ).astype(np.uint8)

    DoG_x = (-b/sigma) * gaus
    DoG_y = (-a/sigma) * gaus
    DoG_x = DoG_x/np.max(DoG_x)
    DoG_y = DoG_y/np.max(DoG_y)

    return DoG_x, DoG_y


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=30)
    # 256x256 sigma 30 의 x,y 미분값들

    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')
    x = ((x - np.min(x)) / np.max(x - np.min(x))).astype(np.float32) # 0~1로 정규화
    y = ((y - np.min(y)) / np.max(y - np.min(y))).astype(np.float32) # 0~1로 정규화
    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = np.sqrt(dst_x*dst_x + dst_y*dst_y)
    cv2.imshow('original',src)
    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)

    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

