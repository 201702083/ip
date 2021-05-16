import cv2
import numpy as np


# jpeg는 보통 block size = 8
def C(w, n=8):
    if w == 0:
        return (1 / n) ** 0.5
    else:
        return (2 / n) ** 0.5


def Spatial2Frequency_mask(block, n=8):
    print(block)
    dst = np.zeros(block.shape)
    v, u = dst.shape

    y, x = np.mgrid[0:u, 0:v]

    mask = np.zeros((n * n, n * n))
    print()
    for v_ in range(v):
        for u_ in range(u):
            t = block*np.cos(((2 * x + 1) * (u_) * np.pi) / (2 * n)) * np.cos(
                ((2 * y + 1) * (v_) * np.pi) / (2 * n))
                # v 0 u 0 -> 1 ??
            mask[v*v_:v*(v_+1),u*u_:u*(u_+1)] = my_normalize(C(v_,n)*C(u_,n)*(t))
            ##########################################################################
            # ToDo                                                                   #
            # mask 만들기                                                             #
            # mask.shape = (16x16)                                                   #
            # DCT에서 사용된 mask는 (4x4) mask가 16개 있음 (u, v) 별로 1개씩 있음 u=4, v=4  #
            # 4중 for문으로 구현 시 감점 예정                                             #
            ##########################################################################
    return mask


def my_normalize(src):
    ##############################################################################
    # ToDo                                                                       #
    # my_normalize                                                               #
    # mask를 보기 좋게 만들기 위해 어떻게 해야 할 지 생각해서 my_normalize 함수 완성해보기   #
    ##############################################################################
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst -= np.min(dst)
    dst /= np.max(dst)
    dst *= 255
    return dst.astype(np.uint8)


if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size))
    # src = my_normalize(src)
    # 4x4 짜리 1
    mask = Spatial2Frequency_mask(src, n=block_size)
    # mask = DCT 마스크
    mask = mask.astype(np.uint8)
    # unsigned int  0 ~ 255 casting

    # 크기가 너무 작으니 크기 키우기 (16x16) -> (320x320)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)
    print(mask)
    cv2.imshow('201702083mask', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
