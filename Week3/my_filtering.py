import cv2
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

def my_filtering(src, ftype, fshape, pad_type='zero'): # 디폴트는 제로패
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
                                # 마스크가 가장자리도 적용될 만큼 패딩 크기 설정

    dst = np.zeros((h, w)) # 반환할 이미지 생성
    average = 1/(fshape[0]*fshape[1])
    mask = np.full(fshape,average)
    cv2.imshow('src', src)
    cv2.imshow('src_pad',src_pad)
    cv2.imshow('dst',dst)
    cv2.imshow('mask',mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

    if ftype == 'average': # 평균 필터링
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################

        #mask 확인
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################
        m = np.zeros(fshape)
        m[fshape[0]//2,fshape[1]//2] = 2
        mask = m - mask
        #mask 확인
        print(mask)

    for i in range(fshape[0]//2, fshape[0]//2+h): # 패딩을 제외한 세로 범위
        for j in range(fshape[1]//2, fshape[1]//2 + w): # 패딩을 제외한 가로 범위


            origin = src_pad[i-fshape[0]//2:(i+fshape[0]//2)+1,j-fshape[1]//2:(j+fshape[1]//2) +1]


            filtervalue = np.sum(origin*mask)
            if(filtervalue < 0) : filtervalue = 0
            if(filtervalue > 255) : filtervalue = 255
            dst[i-fshape[0]//2,j-fshape[1]//2] = filtervalue

    dst = (dst+0.5).astype(np.uint8)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # src 는 원본 (512,512) size
    # repetition padding test
    zer_test = my_padding(src,(20,20))
    rep_test = my_padding(src, (20,20),'repetition')


    # # 3x3 filter
    # dst_average = my_filtering(src, 'average', (3,3))
    # dst_sharpening = my_filtering(src, 'sharpening', (3,3))
    #
    # #원하는 크기로 설정
    # dst_average = my_filtering(src, 'average', (5,7))
    # dst_sharpening = my_filtering(src, 'sharpening', (7,3))

    # # 11x13 filter
    dst_average = my_filtering(src, 'average', (11,13), 'repetition')
    dst_sharpening = my_filtering(src, 'sharpening', (11,13), 'repetition')

    # cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    # cv2.imshow('zero padding test',zer_test.astype(np.uint8))
    # cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()

