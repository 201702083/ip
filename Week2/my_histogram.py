import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):
    src_sh = src.shape
    hist = [0 for a in range(256)] # 0 ~ 255 value 를 index로 갖음 
    for i in range(0, src_sh[0]): # src의 세로 길이 
        for j in range(0, src_sh[1]): # scr 의 가로 길이 
            hist[src[i, j]] += 1 # i,j 픽셀의 value를 참조하여 hist 의 pixel num ++

    plt.plot(hist, color='r')
    plt.title('histogram plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
	# 그래프 출력
    return hist

def my_normalize_hist(histogram, pixel_num): # 정규화
    normalized_hist = [0 for i in range(len(histogram))] # 반환할 리스트 생성
    for i in range(0, len(normalized_hist)): 
        normalized_hist[i] = histogram[i] / pixel_num # histogram의 각 값을 total pixel_num으로 나눔
    return normalized_hist


def my_PDF2CDF(pdf): # 정규화 값 누적
    cdf = pdf # 반환할 리스트 생성
    for i in range(0, len(cdf)): 
        if i > 0:
            cdf[i] = pdf[i] + cdf[i - 1] # cdf는 pdf의 값을 누적한 리스트

    plt.plot(cdf, color='r')
    plt.title('cdf plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
	# 그래프 출력
    return cdf


def my_denormalize(normalized, gray_level): 
    denormalized = normalized # 반환할 리스트 생성 
    for i in range(0 , len(denormalized)): 
        denormalized[i] = denormalized[i] * gray_level # 각 값에 gray_level을 곱함
    return denormalized


def my_calcHist_equalization(denormalized, hist): # 평활화한 histogram 구하기
    hist_equal = [0 for i in range(len(hist))] # 반환할 리스트 0으로 초기화

    for i in range(0, len(denormalized)): 
        hist_equal[denormalized[i]] += hist[i] # floor[i]를 참조한 hist_equal에 hist[i]가누적

    plt.plot(hist_equal, color='r')
    plt.title('histogram_equalization plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
	# 그래프 출력
    return hist_equal


def my_equal_img(src, output_gray_level): # 평활화한 이미지를 반환
    (h,w) = src.shape[:2] # 세로 가로 길이를 받아옴 
    dst = np.zeros((h,w), dtype = np.uint8) # 받아온 크기에 맞춰 zeros 이미지로 생성

    for i in range(h):
        for j in range(w):
            dst[i][j] = output_gray_level[src[i][j]] # Y = Integral 0 to r ( histogram ) 
						     # so 0 ~ src[i][j]의 누적값인 output_gray_level이다

    return dst

def my_round(denormalized_output): # astype(int) 에서 오류가 발생하여 버림 함수 정의
    for i in range(0, len(denormalized_output)):
        denormalized_output[i] = int(denormalized_output[i]) # 각 값을 int 형변환

    return denormalized_output


def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)

    normalized_histogram = my_normalize_hist(histogram, h * w)
    for i in range(len(normalized_histogram)):
        print(normalized_histogram[i],end= ' ')
    print()
    normalized_output = my_PDF2CDF(normalized_histogram)

    denormalized_output = my_denormalize(normalized_output, max_gray_level)

    output_gray_level = my_round(denormalized_output)

    hist_equal = my_calcHist_equalization(output_gray_level, histogram)


    x = range(0,256) # 0 ~ 255
    y = [0 for i in range(len(x))]
    for i in range(0,256):
        y[i] = output_gray_level[i] # Y = Integral 0 to x ( histogram ) = output_gray_level[x]

    plt.plot(x,y)
    plt.title('mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()
	# 그래프 출력
    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal # 평활화한 이미지와 히스토그램 반환

if __name__ == '__main__':
    src = cv2.imread('gapcheon.jpeg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)
    cv2.imwrite('gray.jpg',src)
    cv2.imwrite('sample2.jpg',dst)


    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    cv2.waitKey()
    cv2.destroyAllWindows()
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    cv2.imshow('equalization after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

