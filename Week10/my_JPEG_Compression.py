import numpy as np
import cv2
def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)

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

def my_DCT(src, n = 8):
    (h, w) = src.shape

    h_pad = h + (n - h % n)
    w_pad = w + (n - w % n)

    pad_img = np.zeros((h_pad, w_pad))
    pad_img [:h, :w] = src.copy()

    dst = np.zeros((h_pad, w_pad))

    for row_num in range(h_pad // n):
        for col_num in range(w_pad //n):
            dst[row_num * n : (row_num + 1) * n, col_num * n :(col_num + 1) * n] = \
                get_DCT(pad_img[row_num * n : (row_num + 1) * n, col_num * n :(col_num + 1) * n], n)
    return dst[:h,:w]

def C(w, n = 8):
    if w == 0:
        return (1/n) ** 0.5
    else :
        return (2/n) ** 0.5

def get_DCT(f, n = 8):
    F = np.zeros((n , n))
    for u in range(n):
        for v in range(n):
            x, y = np.mgrid[0:n,0:n]
            val = np.sum(f * np.cos(((2 * x + 1) * u * np.pi)/(2*n)) * np.cos(((2 * y + 1) * v * np.pi)/(2 * n)))
            F[u, v] = C(u, n) * C(v, n) * val

    return F

def get_IDCT(F, n = 8):
    f = np.zeros((n ,n))
    for u in range(n):
        for v in range(n):
            pixel_sum = 0
            # 공식에서 처음부터 다 더하는 연산을 시작하기때문에 값을 저장할 변수 선언
            for row in range(n):
                for col in range(n):
                    # C값을 각 반복문 변수에 넣어주고 주파수 이미지값을 곱하고 코사인값까지 곱해서 값을 누적시킨다.
                    # 거꾸로 진행하기 위해서 기존에서 np.cos(((2 * row + 1) * u * np.pi)/(2*n)) * np.cos(((2 * col + 1) * v * np.pi) / (2 * n)) row값과
                    # u값 col값과 v 값의 위치를 바꿔준다.
                    pixel_sum += C(row) * C(col) * F[row, col] * np.cos(((2 * u + 1) * row * np.pi)/(2*n)) * np.cos(((2 * v + 1) * col * np.pi) / (2 * n))
            f[u, v] = pixel_sum
    return f

def my_IDCT(src, n = 8):
    (h, w) = src.shape
    h_pad = h + (n - h % n)
    w_pad = w + (n - w % n)
    pad_img = np.zeros((h_pad, w_pad))
    pad_img[:h, :w] = src.copy()
    dst = np.zeros((h, w))
    for row_num in range(h// n):
        for col_num in range(w // n):
            dst[row_num * n: (row_num + 1) * n, col_num * n:(col_num + 1) * n] = \
                get_IDCT(src[row_num * n: (row_num + 1) * n, col_num * n:(col_num + 1) * n], n)
    return dst[:h, :w]

def my_JPEG_encoding(src, block_size=8):
    #####################################################
    # TODO                                              #
    # my_block_encoding 완성                            #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # zigzag_value : encoding 결과(zigzag까지)          #
    #####################################################
    (h, w) = src.shape
    src = (src.copy()).astype(np.float)
    # 첫번째 과정 : 원래 이미지에서 128을 빼준다.
    src = src - 128
    # 두번째 과정 : 뺀 이미지에서 DCT 처리를 해준다.
    src = my_DCT(src)
    # 세번째 과정 : Devide Quantization
    n = block_size
    for row in range(h // n):
        for col in range(w //n):
            src[row * n : (row+1) * n, col * n : (col + 1) * n] = np.round(src[row * n : (row+1) * n, col * n : (col + 1) * n] / Quantization_Luminance())
    # 네번째 과정 : zigzag Scanning
    zigzag_value = np.zeros((h * w,))
    # 지그재그 스캐닝을하기위해 1차원 배열로 선언하되 모든 배열의 값이
    # 들어가야하므로 h * w 크기의 1차원 배열을 생성해준다.
    index = 0
    # 인덱스값을 넣기위한 변수
    first = 0
    # 경계값
    for i in range(0, h * w - 1):
        if i < h:
            # 첫번째 행에대한 지그재그행 조건문
            first = 0
        else:
            # 두번째 행부터 마지막 행에 대한 지그재그행 조건문
            first = i - h + 1
        for j in range(first, i - first + 1):
            # 경계값부터 반복문을 돌린다.
            # 지그재그 스캐닝할려면 각 행이 홀수냐 짝수냐를 판별해서 홀수면은 아래로 내려가게 하는 대신에
            # 각 x,y좌표값의 합이 그 행의 값이 되어한다 =>(0,3),(1,2),(2,1),(3,0) 이런식으로
            if i % 2 == 1:
                zigzag_value[index] = src[j, i - j]
            else:
                zigzag_value[index] = src[i - j, j]
            index += 1
    # nan 처리 과정
    # 스캐닝을 완료한 배열을 거꾸로 돌린다.
    zigzag_value = zigzag_value[::-1]
    # 반복문을 통해서 0이 아닌값을 찾아내면 반복문 종료 0인 값은 nan으로 처리해준다.
    for i in range(len(zigzag_value)):
        if zigzag_value[i] != 0:
            break
        else:
            zigzag_value[i] = np.nan
    # 다시 배열을 원상복구 시킨다.
    zigzag_value = zigzag_value[::-1]
    return zigzag_value

def my_JPEG_decoding(zigzag_value, block_size=8):
    #####################################################
    # TODO                                              #
    # my_JPEG_decoding 완성                             #
    # 입력변수는 알아서 설정(단, block_size는 8로 설정)   #
    # return                                            #
    # dst : decoding 결과 이미지                         #
    #####################################################
    # 첫번째 과정 : zigzag decoding
    # 가로 * 세로의 길이만큼의 1차원배열을 2차원으로 형변환해준다.
    zigzag_decoding = np.zeros((int(np.sqrt(len(zigzag_value))), int(np.sqrt(len(zigzag_value)))))

    # nan값을 다 0으로 치환
    for i in range(len(zigzag_value)):
        if np.isnan(zigzag_value[i]):
            zigzag_value[i] = 0
    (h, w) = zigzag_decoding.shape


    dst = np.zeros((h, w))
    index = -1
    # 인덱스 참조 시작 변수 값
    first = 0
    # 경계값 설정
    for i in range(0, h * w - 1):
        if i < h:
            # 첫번째 행에 대한 값을 얻어오기위해 0부터 시작하도록 설정
            first = 0
        else:
            first = i - h + 1
        for j in range(first, i - first + 1):
            index += 1
            # 인덱스 값을 증가시키면서 홀수 번째는 위에서 아래로 내려왔으므로 반대로 해준다.
            if i % 2 == 1:
                zigzag_decoding[j, i - j] = zigzag_value[index]
            else:
                zigzag_decoding[i - j, j] = zigzag_value[index]

    # 두번째 과정 : Quantization 을 곱한다.
    n = block_size
    for row in range(h // n):
        for col in range(w // n):
            zigzag_decoding[row * n: (row + 1) * n, col * n: (col + 1) * n] = \
                np.round(zigzag_decoding[row * n: (row + 1) * n, col * n: (col + 1) * n] * Quantization_Luminance())


    #세번째 과정 : IDCT 처리
    zigzag_decoding = my_IDCT(zigzag_decoding)

    #네번째 과정 : 128을 더한다.
    zigzag_decoding = zigzag_decoding + 128

    zigzag_decoding = zigzag_decoding.astype(np.uint8)

    return zigzag_decoding

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # src = np.array([[52, 55, 61, 66, 70, 61, 64, 73],
    #      [63, 59, 66, 90, 109, 85, 69, 72],
    #      [62, 59, 68, 113, 144, 104, 66, 73],
    #      [63, 58, 71, 122, 154, 106, 70, 69],
    #      [67, 61, 68, 104, 126, 88, 68, 70],
    #      [79, 65, 60, 70, 77, 68, 58, 75],
    #      [85, 71, 64, 59, 55, 61, 65, 83],
    #      [87, 79, 69, 68, 65, 76, 78, 94]])
    # print(src)
    src = src.astype(np.float)
    zigzag_value = my_JPEG_encoding(src)
    #
    dst = my_JPEG_decoding(zigzag_value)
    # dst = my_normalize(dst)
    src = src.astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('result', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


