import cv2
import numpy as np

# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from using_myfunc.my_gaussian import my_filtering
def get_DoG_filter(fsize, sigma=1):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    x = np.mgrid[-(fsize//2):(fsize//2)+1 , ]
    y = x.T
    DoG_x = -(x/(sigma**2))*np.exp(-(x**2+y**2)/(2*(sigma**2)))
    DoG_y = -(y/(sigma**2))*np.exp(-(x**2+y**2)/(2*(sigma**2)))

    return DoG_x, DoG_y
