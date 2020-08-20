# indoorOutdoor
Determines if an image was taken in an indoor or outdoor environment

## Dependences:
numpy, tensorflow, pandas matplotlib

## About
Trained on frames taken from youtube-8m videos using tensorflow.

indoorOutdoor.py trains the model based on the contents of the images directory.
It outputs model.h5

testSingleImage.py will test a single image against the model.
In the commandline, use the -i flag to enter the name of you image

`python testSingleImage.py -f imagename.jpg`
  
