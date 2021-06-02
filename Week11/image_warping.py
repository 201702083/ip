import numpy as np
import cv2

INF = float('inf')


def forward(src, M, fit=False):
    #####################################################
    # TODO                                              #
    # forward 완성                                      #
    #####################################################
    print('< forward >')
    print('M')
    print(M)
    if fit == fit:
        h, w = src.shape
        dot = []
        dot.append(np.dot(M, np.array([[0], [0], [1]])))
        dot.append(np.dot(M, np.array([[w - 1], [0], [1]])))
        dot.append(np.dot(M, np.array([[0], [h - 1], [1]])))
        dot.append(np.dot(M, np.array([[w - 1], [h - 1], [1]])))
        minX = INF
        maxX = -INF
        minY = INF
        maxY = -INF
        for i in range(4):
            if minX > dot[i][0]:
                minX = dot[i][0]
            if maxX < dot[i][0]:
                maxX = dot[i][0]
            if minY > dot[i][1]:
                minY = dot[i][1]
            if maxY < dot[i][1]:
                maxY = dot[i][1]

        dst = np.zeros((int(maxY) - int(minY) + 3, int(maxX) - int(minX) + 3))
        N = np.zeros(dst.shape)

        for row in range(h):
            for col in range(w):

                P = np.array([
                    [col],
                    [row],
                    [1]
                ])
                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0] - minX
                dst_row = P_dst[1][0] - minY
                dst_col_right = int(np.ceil(dst_col))
                dst_col_left = int(dst_col)

                dst_row_bottom = int(np.ceil(dst_row))
                dst_row_top = int(dst_row)
                # if(dst_row_bottom == 296):
                #     print()
                dst[dst_row_top, dst_col_left] += src[row, col]
                N[dst_row_top, dst_col_left] += 1
                # dst_col 이 정수가 아닐 때 우상단 체크
                if dst_col_right != dst_col_left:
                    dst[dst_row_top, dst_col_right] += src[row, col]
                    N[dst_row_top, dst_col_right] += 1
                # dst_row 가 정수가 아닐 때 좌하단 체크
                if dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_left] += src[row, col]
                    N[dst_row_bottom, dst_col_left] += 1
                # dst_col,dst_row 둘 다 정수가 아닐 때 우하단 체크
                if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_right] += src[row, col]
                    N[dst_row_bottom, dst_col_right] += 1

        dst = np.round(dst / (N + 1E-6))
        dst = dst.astype(np.uint8)

    else:
        h, w = src.shape
        print(h, w)
        dst = np.zeros(src.shape)
        N = np.zeros(src.shape)
        for row in range(h):
            for col in range(w):

                P = np.array([
                    [col],
                    [row],
                    [1]
                ])
                P_dst = np.dot(M, P)
                dst_col = P_dst[0][0]
                dst_row = P_dst[1][0]
                dst_col_right = int(np.ceil(dst_col))
                dst_col_left = int(dst_col)

                dst_row_bottom = int(np.ceil(dst_row))
                dst_row_top = int(dst_row)

                # 올림된 정수가 범위 내에 있는 경우에 실행
                if 0 <= dst_col_right < w and 0 <= dst_row_bottom < h:
                    # top, left 를 먼저 채운다. 좌상단 체크
                    dst[dst_row_top, dst_col_left] += src[row, col]
                    N[dst_row_top, dst_col_left] += 1
                    # dst_col 이 정수가 아닐 때 우상단 체크
                    if dst_col_right != dst_col_left:
                        dst[dst_row_top, dst_col_right] += src[row, col]
                        N[dst_row_top, dst_col_right] += 1
                    # dst_row 가 정수가 아닐 때 좌하단 체크
                    if dst_row_bottom != dst_row_top:
                        dst[dst_row_bottom, dst_col_left] += src[row, col]
                        N[dst_row_bottom, dst_col_left] += 1
                    # dst_col,dst_row 둘 다 정수가 아닐 때 우하단 체크
                    if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                        dst[dst_row_bottom, dst_col_right] += src[row, col]
                        N[dst_row_bottom, dst_col_right] += 1

        dst = np.round(dst / (N + 1E-6))
        dst = dst.astype(np.uint8)
    return dst


def backward(src, M, fit=False):
    #####################################################
    # TODO                                              #
    # backward 완성                                      #
    #####################################################
    print('< backward >')
    print('M')
    print(M)
    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)
    if fit == fit:
        h, w = src.shape
        dot = []
        dot.append(np.dot(M, np.array([[0], [0], [1]])))
        dot.append(np.dot(M, np.array([[w - 1], [0], [1]])))
        dot.append(np.dot(M, np.array([[0], [h - 1], [1]])))
        dot.append(np.dot(M, np.array([[w - 1], [h - 1], [1]])))
        minX = INF
        maxX = -INF
        minY = INF
        maxY = -INF
        for i in range(4):
            if minX > dot[i][0]:
                minX = dot[i][0]
            if maxX < dot[i][0]:
                maxX = dot[i][0]
            if minY > dot[i][1]:
                minY = dot[i][1]
            if maxY < dot[i][1]:
                maxY = dot[i][1]

        dst = np.zeros((int(np.ceil(maxY)) - int(np.floor(minY)), int(np.ceil(maxX)) - int(np.floor(minX))))
        h, w = dst.shape
        h_src, w_src = src.shape

        for row in range(h):
            for col in range(w):
                P_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])
                P = np.dot(M_inv, P_dst)
                src_col = P[0][0] + int(minX)
                src_row = P[1][0] + int(minY)
                if 0 < src_col < w_src-1 and 0 < src_row < h_src-1 :
                    src_col_right = int(np.ceil(src_col))
                    src_col_left = int(src_col)
                    src_row_bottom = int(np.ceil(src_row))
                    src_row_top = int(src_row)

                    s = src_col - src_col_left
                    t = src_row - src_row_top

                    intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                                + s * (1 - t) * src[src_row_top, src_col_right] \
                                + (1 - s) * t * src[src_row_bottom, src_col_left] \
                                + s * t * src[src_row_bottom, src_col_right]
                    dst[row , col ] = intensity
        dst = dst.astype(np.uint8)

    else:
        dst = np.zeros(src.shape)
        h, w = dst.shape
        h_src, w_src = src.shape

        for row in range(h):
            for col in range(w):
                P_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])
                P = np.dot(M_inv, P_dst)
                src_col = P[0][0]
                src_row = P[1][0]

                src_col_right = int(np.ceil(src_col))
                src_col_left = int(src_col)
                src_row_bottom = int(np.ceil(src_row))
                src_row_top = int(src_row)

                if src_col_right >= w_src or src_row_bottom >= h_src:
                    continue

                s = src_col - src_col_left
                t = src_row - src_row_top

                intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                            + s * (1 - t) * src[src_row_top, src_col_right] \
                            + (1 - s) * t * src[src_row_bottom, src_col_left] \
                            + s * t * src[src_row_bottom, src_col_right]
                dst[row, col] = intensity
        dst = dst.astype(np.uint8)
    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    #####################################################
    # TODO                                              #
    # M 완성                                             #
    # M_tr, M_sc ... 등등 모든 행렬 M 완성하기              #
    #####################################################
    # translation
    M_tr = np.array([[1, 0, -30], [0, 1, 50], [0, 0, 1]])

    # scaling
    M_sc = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    # rotation
    degree = -np.pi / 9
    M_ro = np.array([[np.cos(degree), -np.sin(degree), 0], [np.sin(degree), np.cos(degree), 0], [0, 0, 1]])

    # shearing
    M_sh = np.array([[1, 0.2, 0], [0.2, 1, 0], [0, 0, 1]])

    # rotation -> translation -> Scale -> Shear
    M = np.dot(M_sh, np.dot(M_sc, np.dot(M_tr, M_ro)))
    # fit이 True인 경우와 False인 경우 다 해야 함.
    fit = True
    # forward
    dst_for = forward(src, M, fit=fit)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    # backward
    dst_back = backward(src, M, fit=fit)
    dst_back2 = backward(dst_back, np.linalg.inv(M), fit=fit)

    cv2.imshow('original', src)
    cv2.imshow('forward1', dst_for)

    cv2.imshow('forward2', dst_for2)
    cv2.imshow('backward1', dst_back)

    cv2.imshow('backward2', dst_back2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
