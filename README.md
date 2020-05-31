# ComputerVision
Learning the ins and outs of computer vision and its various applications 📷

 - [How To Use It](https://github.com/ahmadtc1/ComputerVision#how-to-use-it)
 - [Examples](https://github.com/ahmadtc1/ComputerVision#examples)

# Learning Computer Vision
This repo contains a variety of computer vision techniques and applications I am in the process of exploring. Feel free to download and run any of the files for some computer vision fun yourself!

# How to Use it
First download the required packages using the requirements.txt file
```bash
pip install -r requirements.txt
```

You can start by checking the file's help flag for any additional arguments as shown below

```bash
python detect_faces_video.py -h
```
which will help you identify how to use the file as such
```bash
usage: detect_faces_video.py [-h] --i --m --cI

optional arguments:
  -m, --model         path to the input trained face detection model
  -c, --confidence     desired confidence when detecting faces
  -p, --prototxt      path to the prototxt file associated with the model
```

then allowing you to use the file to execute and see your cool outputs!
```bash
python detect_faces_video.py -c 85 -m f.model -p t.prototxt
```

Be sure to check out the different folders for different cool computer vision applications :)

# Examples
Here's some of the cv applications you can find throughout this repo. Go ahead and try some of em out yourself!

### Test Scanning With Optical Mark Recognition (OMR)
![Optical Mark Recognition](/img/omr.jpg)

### Face Detection
![Face Detection](/img/face_detection.jpg)

### Document Scanning 
![Document Scanner](/img/document_scanner.jpg)

### Object Measurements
![Object Measurer](/img/object_measurements.jpg)