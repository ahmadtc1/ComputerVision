import cv2
import imutils
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="Path to the desired image")

args = vars(ap.parse_args())


#Load image into array
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

#Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Greyscale", gray)
cv2.waitKey(0)


#Apply edge detection to find object outline in the image
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

#Threshold the image, set all pixel vals < 225 to 255. Set all pixel vals >= 225 to 0
#white foregroud, black background
thresh = cv2.threshold(gray, 225,255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Threshed", thresh)
cv2.waitKey(0)


#Detect and draw contours
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
output = image.copy()


for contour in contours:
    cv2.drawContours(output, [contour], -1, (240,0,159), 3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

#Output how many shapes found
text = "I found {} objects.".format(len(contours))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 230, 0), 2)
cv2.imshow("Output", output)
cv2.waitKey(0)

#Apply erosions to reduce foreground object sizes
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)


#Enlarge the foreground regions using dilate
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

#Take our mask and apply a bitwise AND to out input image to only keep the masked regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)