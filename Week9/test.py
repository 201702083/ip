import numpy as np
import cv2

def C(w,n=8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5

def Spatial2Frequency2(block,n=8):
    dst = np.zeros(block.shape)
    v,u = dst.shape

    y,x = np.mgrid[0:u,0:v]

    for v_ in range (v):
        for u_ in range(u):
            tmp = block * np.cos((( 2*x+1)*u_*np.pi)/(2*n)) * np.cos((( 2*y+1)*v_*np.pi)/(2*n))
            print(np.sum(tmp))
            dst[v_,u_] = C(u_, n=n) * C(v_, n=n) * np.sum(tmp)

    dst = np.round(dst,4)

    return dst;
def my_normalize(src):
    ##############################################################################
    # ToDo                                                                       #
    # my_normalize                                                               #
    # mask를 보기 좋게 만들기 위해 어떻게 해야 할 지 생각해서 my_normalize 함수 완성해보기   #
    ##############################################################################
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)
if __name__ == '__main__':
    block_size= 4
    src = np.ones((block_size,block_size))
    print(int(4 // 4))
