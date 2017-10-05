import numpy as np
import cv2

# Read an image from a file and display it to the screen
img = cv2.imread('lena.jpg')
cv2.imshow('input_image', img)

# Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
img_add = img + 2 # This is like increasing brightness
cv2.imshow('add_image', img_add)

img_sub = img - 2 # This is decreasing brightness
cv2.imshow('subtract_image', img_sub)

img_mult = img * 2
cv2.imshow('multiply_image', img_mult)

img_divide = img / 2
cv2.imshow('divide_image', img_divide)

# Resize the image uniformly by 1/2
img_resize = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
cv2.imshow('resize_image', img_resize)

cv2.waitKey(0)
cv2.destroyAllWindows()