import cv2
import numpy as np
from matplotlib.pyplot import imread, imshow

img1 = imread("./linear-algebra-project-farneback-implementation/data/yosemite_sequence/yos2.tif")
img2 = imread("./linear-algebra-project-farneback-implementation/data/ball/image2.tif")

print(img1.shape)
print(img2[:252, :316, 0].shape)

cv2.imwrite('test2.tif',img2[:252, :316, 0])