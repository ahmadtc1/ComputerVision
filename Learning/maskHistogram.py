import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("macOSX")

'''
    Plots a histogram.
    @param {image}  REQUIRED        - an image array **required**
    @param {title}  NOT REQUIRED    - a string 
    @param {mask}   NOT REQUIRED    - a single channel image array 
'''
def plotHistogram(image, title="Histogram", mask=None):
    channels = cv2.split(image)
    colours = ("b", "g", "r")
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (channel, colour) in zip(channels, colours):
        hist = cv2.calcHist([channel], [0], mask, [256], [0, 256])
        plt.plot(hist, color=colour)
        plt.xlim([0, 256])
    
    plt.show()

# Parse user arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Plot the histogram for the original image
plotHistogram(image, "Original Histogram")

# Create a mask to mask the image and display the mask
(centerX, centerY) = (image.shape[1] // 2, image.shape[0] // 2)
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (centerX - 50, centerY - 50), (centerX + 50, centerY + 50), 255, -1)
cv2.imshow("Mask", mask)

# Apply the mask to the image and siaply it
imWithMask = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Image", imWithMask)

# Plot the histogram while using the mask
plotHistogram(image, "Masked Histogram", mask=mask)

cv2.waitKey(0)
