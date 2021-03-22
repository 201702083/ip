import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_calcHist(src):
    src_sh = src.shape
    hist = [0 for a in range(256)]
    for i in range(0,src_sh[0]):
        for j in range(0,src_sh[1]) :
            hist[src[i,j]] +=1

    plt.plot(hist, color = 'r')
    plt.title('histogram plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
    return hist


def my_normalize_hist(histogram, param):
    hist = histogram
    for i in range(0,len(hist)):
        hist[i] = hist[i] / param
    return hist


def my_PDF2CDF(pdf):
    for i in range(0,len(pdf)):
        if i >0 :
            pdf[i] += pdf[i-1]

    plt.plot(pdf, color='r')
    plt.title('cdf plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
    return pdf

def my_denormalize(normalized_output, max_gray_level):
    for i in range(0 , len(normalized_output)):
        normalized_output[i] = normalized_output[i] * max_gray_level
    return normalized_output

def my_calcHist_equalization(floor, hist):
    hist_equal = [0 for i in range(len(hist))]
    for a in range(0,len(hist)):
        print(hist[a],end= ' ')
    print()
    for i in range(0,len(floor)) :
        hist_equal[floor[i]] += hist[i]
        print(floor[i],end='')
        print(' -> ',end='')
        print(hist[i])

    plt.plot(hist_equal, color='r')
    plt.title('histogram_equalization plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
    return hist_equal

def my_equal_img(src, output_gray_level):
    newhist = my_calcHist_equalization(output_gray_level,my_calcHist(src))


def my_round(denormalized_output):
    for i in range(0,len(denormalized_output)):
        denormalized_output[i] = int(denormalized_output[i])
    return denormalized_output

def my_hist_equal (src) :
    (h,w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    print('Histogram')
    for i in range(0, len(histogram)):
        print(histogram[i], end='')
        print(' ', end='')
    print()
    #histogram1 = cv2.calcHist([src],[0],None,[256],[0,256])
    normalized_histogram = my_normalize_hist(histogram, h*w) # 히스토그램 총 픽셀 수로 나눔
    normalized_output = my_PDF2CDF(normalized_histogram) # 나눈 값들을 누적
    denormalized_output = my_denormalize(normalized_output, max_gray_level) # 누적시킨 값에 gray level 곱
    output_gray_level = my_round(denormalized_output) # 버림하여 정수로 변환
    hist = my_calcHist(src)
    hist_equal = my_calcHist_equalization(output_gray_level, hist)
    dst = my_equal_img ( src, output_gray_level )
    return dst, hist_equal

graysrc = cv2.imread('fruits.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('fruits1',graysrc)
my_hist_equal(graysrc)
cv2.waitKey()
cv2.destroyAllWindows()