import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)
    # 홀수 -> 올림, 짝수 -> 버림
    if(scale < 1):
        dst = np.zeros((h_dst, w_dst), dtype=np.uint8)

        for row in range(h_dst):
             for col in range(w_dst):
                 i = row*scale
                 j = col*scale
                 dst[row][col] = src[int(row//scale)][int(col//scale)]
        return dst
    else:
        dst = np.zeros((h_dst+1, w_dst+1), dtype=np.uint8)  # 버림된 1 픽셀을 복구하기 위해 +1을 해준다.

        # bilinear interpolation 적용 # 3x3 -> 15x15
        for row in range(h_dst): # 0~14
            for col in range(w_dst): # 0~14
                i = int(row//scale) # 0 , 1 , 2
                if ( i == h-1): i = i - 1 # 마지막 확대부분에서 인덱스 아웃 처리
                j = int(col//scale) # 0 , 1 , 2
                if ( j == w - 1) : j = j - 1 # ''
                lin_x1 =( (int(src[i][j+1]) - int(src[i][j])) / scale ) * ( col - j*int(scale) ) + src[i][j]
                lin_x2 =( (int(src[i+1][j+1]) - int(src[i+1][j])) / scale ) * (col - j*int(scale) ) + src[i+1][j]
                lin_y =( (( int(lin_x2) - int(lin_x1) )) / scale ) * ( row - i*int(scale) ) + lin_x1
                dst[row][col] = lin_y

                # print('i: ',i,' j: ',j)
                # print('row: ',row,' col: ',col)
                # a( j, src[i][j] ) b( j+1,src[i][j+1])
                # print(int(src[i][j+1]-src[i][j]),' int 밖에 씌움')
                # print( int(src[i][j+1]) - int(src[i][j]), 'int 각각 ')
                # print(int(src[i][j+1]) - int(src[i][j]),' / ',scale, ' * ( ',col,' - ',j*int(scale),' ) + ',src[i][j])
                # if(lin_x1 > 255) : lin_x1 = 255
                # elif(lin_x1<0):lin_x1 =0
                # print(lin_x1)
                # if(lin_x2 > 255) : lin_x2 = 255
                # elif(lin_x2<0):lin_x2 =0
                # print(lin_x2)
                # a( i, lin_x1 ) b( i+1 , lin_x2 )
                # print('x1 : ',lin_x1,' x2: ',lin_x2, ' y: ',lin_y)
                # print()

        return dst

if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    scale = 1/7
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


