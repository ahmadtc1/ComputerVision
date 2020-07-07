from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def imageToFeatureVec(image, size=(32,32)):
    #resize the image to a fixed size and then flatten image
    return cv2.resize(image, size).flatten()

#Handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model file")
args = vars(ap.parse_args())

print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

#Initialize the data matrix and labels list
data = []
labels = []

#Loop over input images
for (i, imagePath) in enumerate(imagePaths):
    #Load image and extract class label
    #Assuming dataset/{class}.{imageNum}.jpg
    image = cv2.imread(imagePath)
    #The first split splits the pathname by slashes (/ or \ depending on the OS) and the indx -1 grabs the filename
    #The second split splits the pathname by the period and grabs either dog or cat

    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    #Construct the feature vector and update the data and labels lists
    features = imageToFeatureVec(image)
    data.append(features)
    labels.append(label)

    #Every 500 images show an update
    if (i > 0 and i % 500 == 0):
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

#Encode the labels by turning them from strings to ints
le = LabelEncoder()
labels = le.fit_transform(labels)

#Scale the pixels of the input image into the range [0,1] and transform the labels into range [0, classesNum]
#this generates a vec for each label where the index of the label is 1 and all other indexes are 0
data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 2)

#Partition the data into training and testing portions
#75% data for training, 25% data for testing
print("[INFO] conducting training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

#Define the network's architecture
model = Sequential()
model.add(Dense(768, input_dim = 3072, init="uniform", activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

#Train the model using SGD (Stochastic Gradient Descent)
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)

#Show the accuracy on the testing set
print("[INFO] evaluating on testing set")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("[INFO]  loss={:.4f}, accuracy:{:.4f}".format(loss, accuracy * 100))

print("[INFO] dumping architecture and weights to file...")
model.save(args["model"])
