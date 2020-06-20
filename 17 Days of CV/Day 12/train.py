import matplotlib as plt
plt.use("Agg")

#import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset")
ap.add_argument("-m", "--model", required=True, help="path to the model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", 
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

#Initialize num of epochs, initial learning rate, batch size and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMENSIONS=(96, 96, 3)

#Init data and labels
data = []
labels = []
#Obtain image paths and shuffle them randomly
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(11)
random.shuffle(imagePaths)

#Loop over the input images
for imagePath in imagePaths:
    #load the image, apply some pre-processing, and store it in the data list
    image = cv2.imread(imagePath)
    image = imutils.resize(image, (IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0]))
    image = img_to_array(image)
    data.append(image)

    #Obtain the class label using the path and update the labels list
    #file structure is dataset/{CLASS_LABEL}/{FILENAME}.jpg
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)