import cv2 as cv
import numpy as np
import argparse


#Parse command line arguments to obtain the video file
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to your input video file for optical flow")
args = vars(ap.parse_args())

# Params for Shi-Tomasi corner detection
featureParams = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)

# Params for Lucas Kanade Optical Flow
lkParams = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Read in the video feed
capture = cv.VideoCapture(args["video"])

# Set the color for drawing flow (red)
color = (0, 0, 255)

ret, firstFrame = capture.read()
previousGray = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)

# Use the Shi Tomasi method to find the strongest corners in the first frame
prev = cv.goodFeaturesToTrack(previousGray, mask = None, **featureParams)

mask = np.zeros_like(firstFrame)

while (capture.isOpened()):
    ret, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Computes sparse optical flow using Lucas-Kanade method
    next, status, error = cv.calcOpticalFlowPyrLK(previousGray, gray, prev, None, **lkParams)

    # selects good feature points for the prev pos (from previousGray frame)
    goodOld = prev[status == 1]

    # selects good feature points for the next pos (our current position)
    goodNew = next[status == 1]

    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        # returns a contiguous flat array as (x, y) for new and old points
        a, b = new.ravel()
        c, d = old.ravel()

        # Draws line between new position and old position
        mask = cv.line(mask, (a, b), (c, d), color, 2)

        # Draws filled circle at new position
        frame = cv.circle(frame, (a, b), 3, color, -1)

    output = cv.add(frame, mask)

    previousGray = gray.copy()

    # Update the previous good feature points
    prev = goodNew.reshape(-1, 1, 2)

    cv.imshow("Sparse Optical Flow", output)

    if (cv.waitKey(10) & 0xFF == ord('q')):
        break

capture.release()
cv.destroyAllWindows