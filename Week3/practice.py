import math

import cv2
import numpy as np
from using_myfunc.padding import my_padding

msize = 5
sigma = 1

x = np.arange(-(msize // 2), (msize // 2) + 1)
y, x = np.mgrid[-(msize // 2):1 + (msize // 2), -(msize // 2):1 + (msize // 2)]

xx = x * x
pad_y=3
pad_x=3
print(xx)
origin = x[0:2,0:2]
for y in range(5):
    for x in range(5):
        origin = xx[(y-1):(y + 1+1), (x - 1):(x + 1+1)]
        print (origin)

# cv2.imshow(src,'src')
# cv2.waitKey()
# cv2.destroyAllWindows()