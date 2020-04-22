import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import argparse
import glob

#Set up an argument parser to have the user input the desired folder from which they would like to edge detect images when executing the script
ap = argparse.ArgumentParser()
ap.add_argument("--i", "--images", required = True,
    help = "Path to set of input images"
)
args = vars(ap.parse_args())

for imagePath in glob.glob(args["i"] + "/*.jpg"):
    try:
        #Load the image as grayscale into an array
        img_array = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        #Gaussian blur the image to remove high noise
        blurred = cv2.GaussianBlur(img_array, (3, 3), 0)

        #Calculate upper and lower threshold values for the image using a statistical formula
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        #Perform edge detection on the image and display it
        edges = cv2.Canny(blurred, lower, upper)
        plt.imshow(edges, cmap='gray')
        plt.show()

    except Exception as e:
        print(e)
        pass

print("Done edge detecting all compatible images")