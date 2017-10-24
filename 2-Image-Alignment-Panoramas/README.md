## Your goal is to create 2 panoramas:
* Using homographies and perspective warping on a common plane (3 images).
* Using affine and cylindrical warping (3 images).

In both options you should:
1. Read in the images: input1.jpg, input2.jpg, input3.jpg
2. [Apply cylindrical wrapping if needed]
3. Calculate the transformation (homography for projective; affine for cylindrical) between each
4. Transform input2 and input3 to the plane of input1, and produce output.png
5. Bonus (!!): Use your Laplacian Blending code to stitch the images together nicely. 
Sample code to do Laplacian Blending with masks [LINK](http://www.morethantechnical.com/2017/09/29/laplacian-pyramid-with-masks-in-opencv-python/).

**TRICY PART:** While doing cylindrical warping we got black boundaries while overlapping, so affine transformed the mask obtained from cylindrically warping the input image. 
Then did alpha blending of affine(cylImage) and affine(mask from cylImg). This removed the black boundaries obtained while blending cylindrically warped images.
Black boundaries were created in between the 2 images because cylindrically warping an image automatically creates black boundaries around the warped image.

## Instructions to run the program:
1. **Perspective Warping:** Creates output image **output_homography.png**
``` python 
python main.py 1 input1.png input2.png input3.png ./
```
2. **Cylindrical Warping:** Creates output image **output_cylindrical.png**
```python
python main.py 2 input1.png input2.png input3.png ./
```
3. **Perspective Warping with Laplacian Blending:** Creates output image **output_homography_lpb.png**
```python
python main.py 3 input1.png input2.png input3.png ./
```
4. **Cylindrical Warping with Laplacian Blending:** Creates output image **output_cylindrical_lpb.png**
```python
python main.py 4 input1.png input2.png input3.png ./
```
5. **RMSD (Root of the Mean Squared Difference):** Professor gave sample output for cases 1 and 2 with water-mark namely example_output1.png for Hommograpy and example_output2.png for Affine.
Expected RMSD was < 20. My RMSD values were around 10. 
Just uncomment RMSD function calls in the main function of main.py and run the file again for any case to print the RMSD values. 
Below are my RMSD values.
* homography 5.98274315887
* homography_bonus 6.10770224727
* affine 9.53223309912
* affine_bonus 9.57650588148
