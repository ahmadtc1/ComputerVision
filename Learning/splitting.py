import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(Blue, Green, Red) = cv2.split(image)

cv2.imshow("Red", Red)
cv2.imshow("Blue", Blue)
cv2.imshow("Green", Green)
cv2.waitKey(0)


merged = cv2.merge([Blue, Green, Red])
cv2.imshow("Merged", merged)
cv2.waitKey(0)

cv2.destroyAllWindows()


zeros = np.zeros(image.shape[:2], dtype="uint8")

cv2.imshow("Red", cv2.merge([zeros, zeros, Red]))
cv2.imshow("Green", cv2.merge([zeros, Green, zeros]))
cv2.imshow("Blue", cv2.merge([Blue, zeros, zeros]))

cv2.waitKey(0)