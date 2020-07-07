from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import time
import dlib
import cv2
import numpy
import imutils

#Calculate the Eye Aspect Ratio (EAR) given coordinates of an eye
#EAR calculated using euclidean distances between certain points
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    eye_aspect_ratio = (A + B) / (2.0 * C)
    return eye_aspect_ratio



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to the facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="optional path to input video")
args = vars(ap.parse_args())

#The Eye Aspect Ratio must be below 0.3 for more than 3 frames to be detected as a blink
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3

#Initialize the ongoing total blink count as well as the consecutive frame count
COUNTER = 0
TOTAL = 0

#Initialize dlib's HOG based face detector then create facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Start the video stream thread
print("[INFO] starting the video stream...")
vs = FileVideoStream(args["video"]).start()
fileStream = False
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    if fileStream and not vs.more():
        #If it's a file video stream and the end of the file is reached, break
        break

    #Read the frame and scale it down as well as convert to grayscale and then extract the faces
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    #Loop over the faces detections
    for rect in rects:
        #Extract the facial landmarks and convert to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #Extract the left and right eye coordinates and calculate the EAR
        leftEye = shape[leftEyeStart: leftEyeEnd]
        rightEye = shape[rightEyeStart: rightEyeEnd]

        EARLeft = eyeAspectRatio(leftEye)
        EARRight = eyeAspectRatio(rightEye)

        #Average the EAR for both eyes
        ear = (EARRight + EARLeft) / 2.0

        #Compute the convex hull for both eyes and visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Increment the frame counter is the EAR is below the blink threshold
        if (ear < EYE_AR_THRESH):
            COUNTER += 1

        #Otherwise the EAR is above the blink threshold
        else:
            #If the eyes have been closed for enough frames, increment the total counts and reset the frame counter
            if COUNTER > EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                COUNTER = 0

        #Place the text on the frame and output it 
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if (key == ord("q")):
            break

#Perform some cleanup
cv2.destroyAllWindows()
vs.stop()