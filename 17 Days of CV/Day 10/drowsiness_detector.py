from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
from threading import Thread
import playsound
import time
import dlib
import numpy as np
import cv2
import argparse
import imutils


def soundAlarm(path):
    #play an alarm sound
    playsound.playsound(path)

def eyeAspectRatio(eye):
    #Calculate the euclidean distance between the different lengths used in EAR calculation
    #Calculate the two vertical distances as well as the horizontal distance
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmarks detector")
ap.add_argument("-a", "--alarm", type=str, default="", help="path to input alarm")
args = vars(ap.parse_args())

#Define constants for the requires AR for a closed eye and required frame for alerting
EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 48

#Initialize a frame counter and a boolean indicating if the alarm is on
COUNTER = 0
ALARM_ON = False

#Initializing dlib's HOG based face detector and landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#Define the starting and ending indexes of the eyes within facial landmarks
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Start the video stream and give the camers 2 seconds to warm up
print("[INFO] starting video stream thread")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Begin the continuous loop to perform ongoing drowsiness detection
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the faces in the picture
    rects = detector(gray, 0)

    #Loop over the detected faces
    for rect in rects:
        #Determine the facial landmarks for the face and extract the left and right eyes
        landmarks = predictor(gray, rect)
        landmarks = face_utils.shape_to_np(landmarks)
        leftEye = landmarks[leftEyeStart: leftEyeEnd]
        rightEye = landmarks[rightEyeStart: rightEyeEnd]

        #Calculate the left and right eye's EAR and average it
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        #Draw the left and right eye contours on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #If the eyes are appearing closed
        if (ear < EYE_AR_THRESH):
            COUNTER += 1

            #If the eyes have been closed for a long enough time
            if (COUNTER > EYE_AR_CONSEC_FRAMES):

                #If the alarm is not already playing
                if not ALARM_ON:
                    ALARM_ON = True

                    #If an alarm file was supplied start a thread to play the alarm sound in the bg
                    if (args["alarm"] != ""):

                        t = Thread(target=sound_alarm,
                        args=(args["alarm"],))
                        t.deamon = True
                        t.start()

                #Place the drowsiness alert text
                cv2.putText(frame, "DROWSINESS ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if (key == ord("q")):
            break
        
            