# indoorOutdoor
Determines if an image was taken in an indoor or outdoor environment

## Dependences:
numpy, tensorflow, pandas, matplotlib

## About
Trained on frames taken from youtube-8m videos using tensorflow.

indoorOutdoor.py trains the model based on the contents of the images directory.
It outputs model.h5, a graph of the prediction loss over the epochs for both the training and validation datasets, it then runs the model on the validation dataset and will let you know the final validation-dataset prediction loss. Will also output all true results / the total number of examples tested.

testSingleImage.py will test a single image against the model.
In the commandline, use the -i flag to enter the name of you image

`python testSingleImage.py -i imagename.jpg`

this will output the percentage the model believes that the image was taken in an indoor or outdoor environment.
