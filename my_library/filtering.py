import numpy as np
import cv2
from my_library.padding import my_padding

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
