import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape # h w 는 Src 의 사이즈
    (p_h, p_w) = pad_shape # p_h,p_w는 패딩할 사이즈
    pad_img = np.zeros((h+2*p_h, w+2*p_w)) # 위아래로 p_h,p_w만큼 늘린 검은색 이미지
    pad_img[p_h:p_h+h, p_w:p_w+w] = src # 가운데에 원본 이미지를 넣음

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up
        pad_img[:p_h, p_w:p_w+w] = src[0,:]
        #down
        pad_img[p_h+h:, p_w:p_w+w] = src[h-1,:]
        #left
        pad_img[:,:p_w] = pad_img[:,p_w:p_w+1]

        #right
        pad_img[:,p_w+w:] = pad_img[:,p_w+w-1:p_w+w]


    else:
        print('zero padding')

    return pad_img