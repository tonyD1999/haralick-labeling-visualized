import numpy as np
import cv2
from haralick import haralick_labeling
from duc_algo import ccl, bfs, dfs

def main():
    img = cv2.imread('images.png')
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
    # ccl(img_bw)
    bfs(img_bw, display=True)
    # dfs(img_bw)


if __name__ == '__main__':
    main()
