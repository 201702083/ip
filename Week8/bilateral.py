import numpy as np
import cv2
import time

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.padding import my_padding
from my_library.filtering import my_filtering
from my_library.my_gaussian import my_get_Gaussian2D_mask

def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):
    #src : 0 ~ 255, dst : 0 ~ 1
    dst = src/255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return my_normalize(dst)

def my_bilateral(src, msize, sigma, sigma_r, pos_x, pos_y, pad_type='zero'):
    ####################################################################################################
    # TODO                                                                                             #
    # my_bilateral 완성                                                                                 #
    # mask를 만들 때 4중 for문으로 구현 시 감점(if문 fsize*fsize개를 사용해서 구현해도 감점) 실습영상 설명 참고      #
    ###################################################################################################

    # 거리에 대한 필터는 모든 마스크가 동일함.
    (h, w) = src.shape
    y, x = np.mgrid[-(msize//2):1+(msize//2),-(msize//2):1+(msize//2)]
    Gmask = (1/(2*np.pi*(2*(sigma**2))))*np.exp(-(x ** 2 + y**2) / (2 * (sigma ** 2)))

    # img_pad : 0 ~ 1
    img_pad = my_padding(src,(msize//2,msize//2),pad_type)

    # 결과 이미지 생성
    dst = np.zeros(src.shape)

    # for 문을 돌며 각 픽셀의 주변 부와의 컬러 차이를 계산하며 각자의 마스크를 계산함.
    for i in range(h):
        print('\r%d / %d ...' %(i,h), end="")
        for j in range(w):
            Colordiff = src[i,j] - img_pad[i+y+msize//2 , j+x+msize//2]       # I(i,j) - I(i+y,j+x) -> 컬러 차이
            color_mask = np.exp(- ( (Colordiff**2)/ (2*(sigma_r**2) ) ) )
            mask = Gmask * color_mask
            mask /= np.sum(mask)
            #mask 의 총합 1

            if i==pos_y and j == pos_x:
                print()
                print(mask.round(4))
                mask_visual = cv2.resize(mask, (200, 200), interpolation = cv2.INTER_NEAREST)
                mask_visual = mask_visual - mask_visual.min()
                mask_visual = (mask_visual/mask_visual.max() * 255).astype(np.uint8)
                cv2.imshow('mask', mask_visual)
                img = img_pad[i:i+5, j:j+5]
                img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_NEAREST)
                img = my_normalize(img)
                cv2.imshow('img', img)
            dst[i, j] = np.sum(mask * img_pad[i:i+5,j:j+5])

    return dst



if __name__ == '__main__':
    start = time.time()
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    np.random.seed(seed=100)

    pos_y = 65
    pos_x = 453
    src_line = src.copy()
    src_line[pos_y-4:pos_y+5, pos_x-4:pos_x+5] = 255
    src_line[pos_y-2:pos_y+3, pos_x-2:pos_x+3] = src[pos_y-2:pos_y+3, pos_x-2:pos_x+3]
    src_noise = add_gaus_noise(src, mean=0, sigma=0.1)
    src_noise = src_noise/255

    ######################################################
    # TODO                                               #
    # my_bilateral, gaussian mask 채우기                  #
    ######################################################
    dst = my_bilateral(src_noise, 5, 3, 0.1, pos_x, pos_y)
    dst = my_normalize(dst)

    gaus2D = my_get_Gaussian2D_mask(5 , sigma = 1)
    dst_gaus2D= my_filtering(src_noise, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src_line)
    cv2.imshow('gaus noise', src_noise)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)
    tital_time = time.time() - start
    print('\ntime : ', tital_time)
    if tital_time > 25:
        print('감점 예정입니다. 코드 수정을 추천드립니다.')
    cv2.waitKey()
    cv2.destroyAllWindows()

