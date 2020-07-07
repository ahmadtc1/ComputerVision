import numpy as np
from keras.models import load_model
from imutils import paths
import imutils
import argparse
import cv2

def image2FeatureVec(image, size=(32,32)):
    #Resize the image to a fixed size and then flatten it
    return cv2.resize(image, size).flatten()

#Handle command line argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to the trained model")
ap.add_argument("-t", "--test-images", required=True, help="path to set of test images")
args = vars(ap.parse_args())

#Initalize class labels for cats and dogs
CLASSES = ["cat", "dog"]

#Load the neural network
print("[INFO] loading the netural network architecture and weights...")
model = load_model(args["model"])
print("[INFO] testing on images in {}".format(args["test_images"]))

for imagePath in paths.list_images(args["test_images"]):
    print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1:]))
    image = cv2.imread(imagePath)
    features = image2FeatureVec(image) / 255.0
    features = np.array([features])

    #Classify the image with the extracted features and pre-trained neural network
    probs = model.predict(features)[0]
    prediction = probs.argmax(axis=0)

    #Draw the identified class and it's probability to the screen
    label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
    cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)