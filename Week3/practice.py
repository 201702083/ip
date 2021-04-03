import math

import cv2
import numpy as np



msize = 5
sigma = 1

x = np.arange(-(msize // 2), (msize // 2) + 1).reshape(1,msize)
print(x)
xx = x*x
print(xx)
print(xx.T)
# cv2.imshow(src,'src')
# cv2.waitKey()
# cv2.destroyAllWindows()