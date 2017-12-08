This repository contains all the assignments done as part of CSE527 - Computer Vision class done under [Professor Roy Shilkrot](http://hi.cs.stonybrook.edu/). Below is the summary of tasks in each assignment.

## 0. Hello Vision
Write an OpenCV program to do the following things:
* Read an image from a file and display it to the screen
* Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
* Resize the image uniformly by ½
  
## 1. Histograms, Filters, Deconvolution, Blending
* Perform Histogram Equalization on the given input image.
* Perform Low-Pass, High-Pass and Deconvolution on the given input image.
* Perform Laplacian Blending on the two input images (blend them together).

## 2. Image Alignment, Panoramas
Your goal is to create 2 panoramas:
* Using homographies and perspective warping on a common plane (3 images).
* Using cylindrical warping (many images).

## 4. Segmentation
Your goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:

## 5. Structured Light [3D Reconstruction]
Your goal is to reconstruct a scene from multiple structured light scannings of it.

### 6. CNNs and Transfer Learning using TensorFlow
Your goal is to 
* Train an MNIST CNN classifier on just the digits: 1, 4, 5 and 9
* Use your trained model’s weights on the lower 4 layers to train a classifier for the rest of MNIST (excluding 1,4,5 and 9)
  * Try to run as few epochs as possible to get a good classification (> 99% on test)
  * Try a session with freezing the lower layers weights, and also a session of just fine-tuning the weights.
