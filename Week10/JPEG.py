import numpy as np
import cv2
import time

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    ######################################
    h,w = src.shape
    pad_need_h = h % n
    pad_need_w = w % n

    pad_img = np.zeros((h + pad_need_h, w + pad_need_w))
    pad_img[:h,:w] = src.copy()
    h,w = pad_img.shape
    blocks = []
    for i in range(h//n):
        i *= n
        for j in range(w//n):
            j *=n
            block = pad_img[i:i+n,j:j+n]
        blocks.append(block)
    return np.array(blocks)

def C(w, n=8):
    if w == 0:
        return (1 / n) ** 0.5
    else:
        return (2 / n) ** 0.5

def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    x, y = np.mgrid[0:n, 0:n]
    dst = np.zeros((n , n))
    for u in range(n):
        for v in range(n):
            val = np.sum(block * np.cos(((2 * x + 1) * u * np.pi)/(2*n)) * np.cos(((2 * y + 1) * v * np.pi)/(2 * n)))
            dst[u, v] = C(u, n) * C(v, n) * val
    return np.round(dst)
def searchEOB(dst,eob, i):
    if ( i == 0 ): eob.insert(len(eob),i)
    else:
        dst.extend(eob)
        dst.insert(len(dst),i)
        eob = []
    return dst,eob
def my_zigzag_scanning(blockQ , mode ='encoding', block_size = 8):
    ######################################
    # TODO                               #
    # my_zigzag_scanning 완성             #
    ######################################

    #QuT 된 블럭을 지그재그로 스캔한다.
    dst = []
    eob = []
    i,j = 0 , 0
    if ( mode == 'encoding'):
        while (1) :
            dst,eob = searchEOB(dst,eob,blockQ[i][j])
            if(i == blockQ.shape[0]-1) & (j == blockQ.shape[1]-1): break
            # 끝점 도달 시 탈출
            if ( j < blockQ.shape[1]-1):
                j +=1
            # 우측으로 이동가능하면 우측으로 이동
            else :
                i + 1
            # 그렇지 않다면 아래로 이동
            while(1):
                if ( i == blockQ.shape[0]-1) | ( j == 0):break
                #왼쪽 or 아래 가장자리 도달 시 탈출
                dst, eob = searchEOB(dst, eob, blockQ[i][j])
                i +=1
                j -=1
            #탈출 후 끝점 탐색
            dst,eob = searchEOB(dst,eob,blockQ[i][j])

            if ( i < blockQ.shape[0]-1):
                i +=1
            else :
                j +=1
            while(1):
                if( i == 0) | ( j == blockQ.shape[1]-1):break
                # 오른쪽 or 위 가장자리 도달 시 탈출
                # 여기서 끝점 제외 전부 탐색
                dst, eob = searchEOB(dst, eob, blockQ[i][j])
                i-=1
                j+=1
            # 탈출 후 끝점 탐색

        # 마지막 인덱스 도달 시 해당 인덱스도 탐색해야함
        dst,eob = searchEOB(dst,eob,blockQ[i][j])
        if ( len(dst) < (blockQ.shape[0] * blockQ.shape[1])):
            eob = ['EOB']
            dst.extend(eob)


    else : # 디코딩 모드 zigzag 스캐닝이 된 블록이 들어온다.
        if (len(blockQ) == block_size): #EOB 가 없는 경우
            return blockQ
        else: # EOB가 있는 경우
            blockQ.pop(len(blockQ)-1) # EOB 를 제거
            dst = np.zeros((block_size,block_size))
            count = 0
            # 1개씩 넣을 때 마다 +1 해줄 것
            while ( count < len(blockQ)):
                dst[i,j] = blockQ[count]
                count +=1
                if (j < dst.shape[1] - 1):
                    j += 1
                # 우측으로 이동가능하면 우측으로 이동
                else:
                    i + 1
                # 그렇지 않다면 아래로 이동
                while (count < len(blockQ)):
                    if (i == dst.shape[0] - 1) | (j == 0) : break
                    # 왼쪽 or 아래 가장자리 도달 시 탈출
                    dst[i, j] = blockQ[count]
                    count += 1
                    i += 1
                    j -= 1
                if( count == len(blockQ)): break
                dst[i, j] = blockQ[count]
                count += 1
                if (i < dst.shape[0] - 1):
                    i += 1
                else:
                    j += 1
                while (count < len(blockQ)):
                    if (i == 0) | (j == dst.shape[1] - 1) : break
                    # 오른쪽 or 위 가장자리 도달 시 탈출
                    dst[i, j] = blockQ[count]
                    count += 1
                    i -= 1
                    j += 1


    return dst

def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    x, y = np.mgrid[0:n, 0:n]
    dst = np.zeros((n , n))

    # C(u)C(v) 과정
    block[1:,:n] *= np.sqrt(2)
    block[:n,1:] *= np.sqrt(2)
    block /= n

    for u in range(n):
        for v in range(n):
            val = np.sum(block*np.cos(((2 * u + 1) * x * np.pi)/(2*n)) * np.cos(((2 * v + 1) * y * np.pi)/(2 * n)))
            dst[u, v] = val
    return np.round(dst)

def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    q,h,w = blocks.shape
    # Lena => q = 64 , h w = 8,8 / 8x8 array가 64 개 있음
    #src_shape 는 원본 크기 blocks은 더 클 수도 있음. 모자란 부분을 패딩해줬기 때문
    dst = np.zeros(src_shape)
    c = 0
    dst[0:8,0:8] = blocks[1]
    print(blocks[c])
    print(dst)
    # for  i in range (512//n):
    #     for j in range( 512//n):
    #         dst[i*n:n*(i+1), j*n:n*(j+1)] = blocks[c]
    #         c +=1


    return dst

def Encoding(src, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)

    #subtract 128
    blocks -= 128
    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)

    # zigzag scanning
    zz = []
    #QnT 는 각 블럭들을 Thresh, quanti 한
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))
    # zz는 list * list
    return zz, src.shape

def Decoding(zigzag, src_shape, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    # add 128
    blocks_idct += 128

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst



def main():
    start = time.time()
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    comp, src_shape = Encoding(src, n=8)
    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 2줄은 주석처리하고, 아래 2줄은 주석 풀기)
    #comp = np.load('comp.npy', allow_pickle=True)
    #src_shape = np.load('src_shape.npy')

    recover_img = Decoding(comp, src_shape, n=8)
    total_time = time.time() - start
    print('time : ', total_time)
    if total_time > 45:
        print('감점 예정입니다.')
    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
