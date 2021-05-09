import cv2
import numpy as np
from my_library.DoG import *
from my_library.my_gaussian import my_filtering
from my_library.padding import my_padding


# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################
    Ix, Iy = get_DoG_filter(fsize, sigma)
    return my_filtering(src, Ix, 'repetition'), my_filtering(src, Iy, 'repetition')


# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.abs(Ix) + np.abs(Iy)
    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):  #
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    angle = np.arctan(Iy / (Ix + e))
    return angle


# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    # angle = -90 ~ +90
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    # largest_magnitude = np.zeros(magnitude.shape)
    # 제로패딩해서 가장자리 부분도 supression이 가능하게
    largest_magnitude = np.zeros(magnitude.shape)

    dst = my_padding(magnitude, (1, 1), 'zero').astype(np.float64)  # size ( h+2, w+2 )
    # magnitude 의 모든 픽셀을 돌며 검사
    h, w = magnitude.shape
    for row in range(h):
        for col in range(w):
            # 각 픽셀의 엣지와 수직인 각도인 angle을 통해 case 를 나눈다.
            grad = angle[row][col]  # -ㅠ/2 ~ + ㅠ/2
            d_r = row + 1
            d_c = col + 1
            if (grad < - (np.pi / 4)):
                f1 = (1 / np.tan(grad)) * dst[d_r - 1][d_c + 1] + (1 - (1 / np.tan(grad))) * dst[d_r - 1, d_c]
                f2 = (1 / np.tan(grad)) * dst[d_r + 1][d_c - 1] + (1 - (1 / np.tan(grad))) * dst[d_r + 1, d_c]
            elif (grad < 0):
                f1 = np.tan(grad) * dst[d_r - 1][d_c + 1] + (1 - np.tan(grad)) * dst[d_r, d_c + 1]
                f2 = np.tan(grad) * dst[d_r + 1][d_c - 1] + (1 - np.tan(grad)) * dst[d_r, d_c - 1]
            elif (grad < np.pi / 4):
                f1 = np.tan(grad) * dst[d_r + 1][d_c + 1] + (1 - np.tan(grad)) * dst[d_r, d_c + 1]
                f2 = np.tan(grad) * dst[d_r - 1][d_c - 1] + (1 - np.tan(grad)) * dst[d_r, d_c - 1]
            elif (grad <= np.pi / 2):
                f1 = (1 / np.tan(grad)) * dst[d_r + 1][d_c + 1] + (1 - (1 / np.tan(grad))) * dst[d_r + 1, d_c]
                f2 = (1 / np.tan(grad)) * dst[d_r - 1][d_c - 1] + (1 - (1 / np.tan(grad))) * dst[d_r - 1, d_c]
            f3 = magnitude[row][col]

            if (f3 >= f1) & (f3 >= f2):
                largest_magnitude[row][col] = f3

    return largest_magnitude


# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy()

    # dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)

    (h, w) = dst.shape
    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################
    ret, dst_high = cv2.threshold(dst, high_threshold_value, 255, cv2.THRESH_BINARY)
    ret, dst_row = cv2.threshold(dst, low_threshold_value, 127, cv2.THRESH_BINARY)
    high_pad = my_padding(dst_high, (1, 1), 'zero')  # size ( h+2, w+2 )
    cv2.imshow('double',cv2.add(dst_row,dst_high))
    change = 1
    while(change != 0):
        change = 0
        for row in range ( h  ):
            for col in range ( w ) :
                high = high_pad[row:row+2 , col : col+2]
                if ( np.sum(high) > 0) & (dst_row[row][col] == 127):
                    dst_row[row][col] = 255
                    high_pad[row+1][col+1] = 255
                    change = 1
                elif dst_row[row][col] == 255:
                    dst_row[row][col] = 255
                elif dst_row[row][col] == 127:
                    dst_row[row][col] = 127

    for row in range ( h  ):
        for col in range ( w ) :
            if(dst_row[row][col] == 127) : dst_row[row][col] = 0

    return dst_row


def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
