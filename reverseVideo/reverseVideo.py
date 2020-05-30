import argparse
import cv2
import imutils
from imutils.video import FileVideoStream
import numpy
import time
import logging

#Set up basic logging for the application
logging.basicConfig(filename='reverseVideo.log', filemode='w', level=logging.INFO)

#Parse command line arguments to obtain the video file
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to your input video file to be reversed")
ap.add_argument("-c", "--codec", type=str, default="mp4v", help="codec of output video")
args = vars(ap.parse_args())

logging.info("Using FourCC codec: %s", args["codec"])

#Create the file videoStream and sleep for 2 seconds to let it warm up
vs = FileVideoStream(args["video"]).start()
logging.info("Starting file video stream")

#initialize some variables for storing and using data
frameStack = []
output = "reversed_" + args["video"]
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
(h, w) = (0, 0)

while True:
    #If we reached the end of the file break from the loop
    if not vs.more():
        logging.info("End Of File reached. Done extracting frames")
        break

    frame = vs.read()
    if frame is None:
        logging.debug("Read \"None\" frame while reading video file. Ignoring this frame")
        continue

    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    frameStack.append(frame)

    cv2.imshow("frame", frame)
    cv2.waitKey(1) & 0xFF

logging.info("Extracted %d frames", len(frameStack))
logging.info("Beginning to display video in reverse and write reversed video file")
time.sleep(1)


writer = cv2.VideoWriter(output, fourcc, 20, (w, h), True)

#Loop through the obtained frames and display them in a LIFO manner, reversing the video
while (len(frameStack) > 0):
    frame = frameStack.pop()
    writer.write(frame)
    cv2.imshow("reversed",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

logging.info("Done displaying reverse video and writing frames. Now performing cleanup")
writer.release()
cv2.destroyAllWindows()