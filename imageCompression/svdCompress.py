import cv2
import os
import numpy as np
import argparse

#Parse arguments from the user
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="input image to be compressed", required=True)
ap.add_argument("-r", "--r", help="desired r value for svd compression", default=100, required=False)
args = vars(ap.parse_args())

#Load the image and convert it to grayscale
img = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("[INFO] Converted colour from bgr to grayscale")

#use numpy svd to extract the U, S, and VT matrices from the image and operate on the S matrix
U, S, VT = np.linalg.svd(img, full_matrices=False)
S = np.diag(S)
print("[INFO] Extracted svd matrices from image matrix")
r = args["r"]

#Generate the compressed image array
Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r, :]
print("[INFO] created compressed grayscale image")

#Create a filename for the compressed file to be written to disk
fileName = str(args["image"])
fileName = fileName[:fileName.rfind(".")] + "_compressed.jpg"

cv2.imwrite(fileName, Xapprox)
