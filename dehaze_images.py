import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dark_channel(img, wind_size):
    no_rows = img.shape[0]
    no_cols = img.shape[1]
    min_channel = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    dark_channel = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for row in range(no_rows):
        for col in range(no_cols):
            min_channel[row][col] = np.min(img[row,col,:])
    for row in range(wind_size//2, no_rows-wind_size//2):
        for col in range(wind_size//2, no_cols-wind_size//2):
            dark_channel[row][col] = np.min(min_channel[row-wind_size//2:row+wind_size//2,col-wind_size//2:col+wind_size//2])
    return dark_channel


img = cv2.imread("../images/pic2.png")
dark_channel_img = get_dark_channel(img, 15)
cv2.imshow('image',np.uint8(dark_channel_img))
cv2.waitKey(0)
cv2.destroyAllWindows()

