import cv2
import numpy as np
import imutils

image = cv2.imread("jp.png")

(height, width, channels) = image.shape
print("width: {}, height: {}, channels: {}".format(width, height,channels))

cv2.imshow("Image", image)

cv2.waitKey(0)

(blue, red, green) = image[100,50]
print("R={}, G={}, B={}".format(red, green, blue))

#Extract a 100x100 region of interest (ROI) from the image
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

resized = cv2.resize(image, (200,200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

#Scaling down image size with respect to the aspect ratio
scalingFactor = 300.0/width
dimensions = (300, int(height * scalingFactor))
resized = cv2.resize(image, dimensions)

cv2.imshow("Scaled Resized Image", resized)
cv2.waitKey(0)


#Try using imutils resize function so we dont have to manually resize
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)


#Rotate image manually
center = (width // 2, height // 2)
rotationMatrix = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, rotationMatrix, (width, height))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)


#Rotating image using imutils
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)


#Rotate image while keeping bounds using imutils
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Rotate Bounded", rotated)
cv2.waitKey(0)


#Gaussian Blur
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

#Draw a rectangle
output = image.copy()
cv2.rectangle(output, (320,60), (420,160), (150,100,255), 3)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

#Draw a circle
output = image.copy()
cv2.circle(output, (300,150), 20, (255,0,0), 6)
cv2.imshow("Circle", output)
cv2.waitKey(0)

#Draw a line
output = image.copy()
cv2.line(output, (60,20), (400,200), (0,0,255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

#Place text on an image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park", (10,25),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imshow("Text", output)
cv2.waitKey(0)