import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape # h w 는 Src 의 사이즈
    (p_h, p_w) = pad_shape # p_h,p_w는 패딩할 사이즈
    pad_img = np.zeros((h+2*p_h, w+2*p_w),dtype=np.uint8) # 위아래로 p_h,p_w만큼 늘린 검은색 이미지
    pad_img[p_h:p_h+h, p_w:p_w+w] = src # 가운데에 원본 이미지를 넣음

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up
        for i in range(0,p_h):
            for j in range(p_w,w+p_w):
                pad_img[i,j] = pad_img[p_h,j]
        #down
        for i in range(p_h+h, 2*p_h+h):
            for j in range(p_w, w + p_w ):
                pad_img[i, j] = pad_img[p_h+h-1, j]

        #left
        for i in range(0,h+2*p_h):
            for j in range(0,p_w):
                pad_img[i,j] = pad_img[i,p_w]
        #right
        for i in range(0, h + 2 * p_h):
            for j in range(w+p_w, w+2*p_w):
                pad_img[i, j] = pad_img[i, w+p_w -1]

    else:
        print('zero padding')

    return pad_img