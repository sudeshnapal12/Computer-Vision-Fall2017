# Your goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:
1. Given an image and sparse markings for foreground and background
2. Calculate SLIC over image
3. Calculate color histograms for all superpixels
4. Calculate color histograms for FG and BG
5. Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
6. Run a graph-cut algorithm to get the final segmentation

### How to run?
```
>> python main.py astronaut.png astronaut_marking.png ./
```
Segmented to stored in 'mask.png'.
 

# "Wow factor" bonus (Interactive segmentation):
Implement an interactive interface with OpenCV so that we can draw the marking with our mouse instead of using the given marking. 
1. Make it interactive: Let the user draw the markings (carrying 0 pt for this part)
2. for every interaction step (mouse click, drag, etc.)
    * recalculate only the FG-BG histograms,
    * construct the graph and get a segmentation from the max-flow graph-cut,
    * show the result immediately to the user (should be fast enough).
    
### How to run?
```
>> python main_bonus.py astronaut.png astronaut.png ./
```

* Image window opens displaying an image to be segmented. Red is for foreground and Blue is for background.
* Key press 'm' to toggle between red and blue.
* After a minimum one marking each of foreground and background, segmented image window pops up.
* Subsequent blue or red markings update the segmented image accordingly.
* Key press 'ESC' saves the segmented image in 'mask.png' and closes all the image windows.
