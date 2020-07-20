import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("macOSX")

# Parse user arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform histogram equalization on the grayscaled image
equalized = cv2.equalizeHist(gray)

cv2.imshow("Histogram Equalization", np.hstack([gray, equalized]))

# Calculate the histogram of both the original and equalized images
hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])
origHist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plot the histogram for the equalized histogram
plt.figure()
plt.title("Hist Equalized Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

# Plot the histogram for the non-equalized histogram
plt.figure()
plt.title("Non Equalized Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(origHist)
plt.xlim([0, 256])
plt.show()

cv2.waitKey(0)