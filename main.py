import numpy as np
import cv2
from haralick import haralick_labeling
from duc_algo import ccl, bfs, dfs
import sys

def main():
    img = cv2.imread(sys.argv[1])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img_bw = np.array([[1,0,0,0,0,0,0,0,0,0],
    #                    [0,1,1,0,1,1,1,0,1,0],
    #                    [0,1,1,0,1,0,1,0,1,0],
    #                    [0,1,1,1,1,0,0,0,1,0],
    #                    [0,0,0,0,0,0,0,0,1,0],
    #                    [0,1,1,1,1,0,1,0,1,0],
    #                    [0,0,0,0,1,0,1,0,1,0],
    #                    [0,1,1,1,1,0,0,0,1,0],
    #                    [0,1,1,1,1,0,1,1,1,0],
    #                    [0,0,0,0,0,0,0,0,0,0]])
    # haralick_labeling(img_bw, display=True)
    if sys.argv[2] == 'hara':
        haralick_labeling(img_bw, display=True)
    elif sys.argv[2] == 'twopass4':
        ccl_4(img_bw, display=True)
    elif sys.argv[2] == 'twopass8':
        ccl_8(img_bw, display=True)
    elif sys.argv[2] == 'bfs':
        bfs(img_bw, display=True)
    elif sys.argv[2] == 'dfs':
        dfs(img_bw, display=True)


if __name__ == '__main__':
    main()
