# Implementation of Alexnet in Tensorflow

## This code is based on Alexnet architecture and from `ImageNet Classification with Deep Convolutional Neural Networks` paper by Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton.

Input must be a 227 X 227 X 3 size image. If we want to work on other size images (like German Traffic Sign database which has 32 X 32 X 3 image size) then the first step is to reshape those images to 227 X 227. 

`Alexnet` method takes input features(image), weights and biases(which are pretrained and we can get the file `bvlc-alexnet.npy` online). `Feature_extract` switch is used for Feature extraction. When it is set to False, it returns probabilities of classes and when it is set to True it will return 7th layer (2nd Fully Connected layer) tensor. Next steps are to freeze all the layers before it, add the last fully connected layer according to new dataset classes and then train that last layer only.