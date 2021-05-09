import numpy as np
import cv2
from using_myfunc.padding import my_padding

def my_filtering(src, filter, pad_type = 'zero'):
    h,w = src.shape
    f_h,f_w = filter.shape
    src_pad = my_padding(src,(f_h//2,f_w//2),pad_type)
    dst = np.zeros((h,w))

    for row in range(h):
        for col in range(w):
            val = np.sum(src_pad[row:row+f_h, col:col+f_w] * filter )
            val = np.clip(val,0,255)
            dst = [row,col] = val

    dst = (dst+0.5).astype(np.uint8)
    return dst

