import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_gaussian import my_filtering
def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-(fsize//2):1+(fsize//2),-(fsize//2):1+(fsize//2)]

    # DoG_x = -(x / (sigma ** 2)) * (np.math.e ** (- (x * x + y * y) / (2 * sigma * sigma)))
    # DoG_y = -(y / (sigma ** 2)) * (np.math.e ** (- (x * x + y * y) / (2 * sigma * sigma)))

    DoG_x = (-x * (np.math.e ** (-(x*x+y*y) / (2*sigma*sigma))))
    DoG_y = (-y * (np.math.e ** (-(x*x+y*y) / (2*sigma*sigma))))

    # DoG_x = ((DoG_x - np.min(DoG_x)) / np.max(DoG_x - np.min(DoG_x)) * 255).astype(np.uint8)
    # DoG_y = ((DoG_y - np.min(DoG_y)) / np.max(DoG_y - np.min(DoG_y)) * 255).astype(np.uint8)

    return DoG_x, DoG_y


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    print(src)
    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=30)
    # 256x256 sigma 30 의 x,y 미분값들

    dst_x = my_filtering(src, x, 'zero')
    dst_y = my_filtering(src, y, 'zero')
    x = (x - np.min(x)) / np.max(x - np.min(x))
    y = (y - np.min(y)) / np.max(y - np.min(y))
    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    # dst = np.sqrt(dst_x*dst_x + dst_y*dst_y)
    cv2.imshow('original',src)
    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_x z ',dst_x)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst_y z ',dst_y)

    # cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

