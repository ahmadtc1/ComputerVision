import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
import imutils
import time


#Argument Parsing
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'Deploy' prototxt file")

ap.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")

ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probbability to filter weak detections")

args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


#Start the video stream and let it sleep for 2 seconds to allow it to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:

    #Obtain the frame from the video stream
    frame = vs.read()
    #Resize the frame to be 400 px wide
    frame = imutils.resize(frame, width=400)

    #Extract the height and width of the the frame
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        #Obtain the confidence from the detections
        confidence = detections[0,0,i,2]

        #If the confidence is above our minimum threshold, show the detected face
        if (confidence < args["confidence"]):
            continue

        #Find the x and y bounds for the bounding box representing the found face
        box = detections[0,0,i,3:7] * np.array([width,height,width,height])
        (startX, startY, endX, endY) = box.astype("int")


        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        #Prepare and output the the frame with the now bounded face along with the confidence
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
        cv2.putText(frame, text, (startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
        cv2.imshow("Frame", frame)

        #Query a keypress and quit if q is pressed
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('q')):
            break

#Clean up the cv2 windosw and the video stream
cv2.destroyAllWindows()
vs.stop()

