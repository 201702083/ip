import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):
    src_sh = src.shape
    hist = [0 for a in range(256)]
    for i in range(0, src_sh[0]):
        for j in range(0, src_sh[1]):
            hist[src[i, j]] += 1

    plt.plot(hist, color='r')
    plt.title('histogram plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
    return hist

def my_normalize_hist(histogram, pixel_num):
    normalized_hist = [0 for i in range(len(histogram))]
    for i in range(0, len(normalized_hist)):
        normalized_hist[i] = histogram[i] / pixel_num
    return normalized_hist


def my_PDF2CDF(pdf):
    cdf = pdf
    for i in range(0, len(cdf)):
        if i > 0:
            cdf[i] = pdf[i] + pdf[i - 1]

    plt.plot(cdf, color='r')
    plt.title('cdf plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
    return cdf


def my_denormalize(normalized, gray_level):
    denormalized = normalized
    for i in range(0 , len(denormalized)):
        denormalized[i] = denormalized[i] * gray_level
    return denormalized


def my_calcHist_equalization(denormalized, hist):
    hist_equal = [0 for i in range(len(hist))]

    for i in range(0, len(denormalized)):
        hist_equal[denormalized[i]] = hist[i]

    plt.plot(hist_equal, color='r')
    plt.title('histogram_equalization plot')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
    return hist_equal


def my_equal_img(src, output_gray_level):
    (h,w) = src.shape[:2]
    dst = np.zeros((h,w), dtype = np.uint8)

    for i in range(h):
        for j in range(w):
            dst[i][j] = output_gray_level[src[i][j]]

    return dst

def my_round(denormalized_output):
    for i in range(0, len(denormalized_output)):
        denormalized_output[i] = int(denormalized_output[i])

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


    x = range(0,256)
    y = [0 for i in range(len(x))]
    for i in range(0,256):
        y[i] = output_gray_level[i]

    plt.plot(x,y)
    plt.title('mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

if __name__ == '__main__':
    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    cv2.waitKey()
    cv2.destroyAllWindows()
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    plt.figure(figsize=(8, 5))
    cv2.imshow('equalizetion after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

