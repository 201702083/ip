import numpy as np
import cv2
import matplotlib.pyplot as plt
()


def my_calcHist_gray_mini_img(mini_img):
    h,w = mini_img.shape[:2] # 가로세
    hist = np.zeros((10,),dtype = np.int8)
    cv2.imshow('gg',hist)
    for row in range(h):
        for col in range(w):
            intensity = mini_img[row,col]
            hist[intensity]+=1
    return hist



if __name__ == '__main__':
    src = np.array([[3,1,3,5,4],[9,8,3,5,6],
                    [2,2,3,8,7],[5,4,6,5,4],
                    [1,0,0,2,6]], dtype = np.uint8)
    src1 = cv2.imread('fruits.jpg')
    src1 = src1[0:100,:0:100]
    hist = my_calcHist_gray_mini_img(src)
    binX = np.arange(len(hist))
    plt.bar(binX, hist, width = 0.8, color = 'g')
    plt.title('histogram')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()
quit()
cv2.waitKey()
cv2.destroyAllWindows()