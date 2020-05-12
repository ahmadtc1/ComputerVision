import numpy as np
import argparse
import time
import cv2

#Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help='path to Caffe \'deploy\' prototxt file')
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels")
args = vars(ap.parse_args())

#load the image
image = cv2.imread(args["image"])

#Load the class labels
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#CNN required fixed spatial dimensions so must ensure that image is 224x224
#Used resized image to perform mean subtraction (104, 117, 123) to normalize input
#After the command out blob has the shape (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

#Load the model
print("[INFO] loading our serialized model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Set blob as input to network and do a forward-pass to obtain output classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

#Sort the probability indexes in descending order (most likely prediction first) and grab the top 5 predictions
indexes = np.argsort(preds[0])[::-1][:5]

#Loop over the top 5 predictions
for (i, index) in enumerate(indexes):
    #Draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%.".format(classes[index], preds[0][index] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print("[INFO] {}. label {}, probability: {:.5}".format((i + 1), classes[index], preds[0][index]))

cv2.imshow("Predicted", image)
cv2.waitKey(0)