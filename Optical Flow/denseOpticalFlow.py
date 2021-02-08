import cv2
import numpy as np
import argparse

#Parse command line arguments to obtain the video file
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to your input video file for optical flow")
args = vars(ap.parse_args())

capture = cv2.VideoCapture(args["video"])

ret, firstFrame = capture.read()
previousGray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(firstFrame)
mask[..., 1] = 255

while (capture.isOpened()):
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("Dense Optical Flow", rgb)

    previousGray = gray
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

capture.release()
cv2.destroyAllWindows()