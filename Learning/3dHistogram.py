from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import matplotlib

matplotlib.use("macOSX")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-s", "--size", required=False, help="size of largest colour bin", default=5000)
ap.add_argument("-b", "--bins", required=False, help="number of bins per colour channe", default=8)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
size = float(args["size"])
bins = int(args["bins"])

hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

print("3D Histogram Shape: %s,  with values %d".format(hist.shape, hist.flatten().shape[0]))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ratio = size / np.max(hist)

for (x, plane) in enumerate(hist):
    for (y, row) in enumerate(plane):
        for (z, col) in enumerate(row):
            if hist[x][y][z] > 0.0:
                siz = ratio * hist[x][y][z]
                rgb = (z / (bins - 1) , y / (bins - 1), x / (bins - 1))
                ax.scatter(x, y, z, s=siz, facecolors=rgb,)

plt.show()
cv2.waitKey(0)