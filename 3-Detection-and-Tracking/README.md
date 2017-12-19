# Detection and Tracking
* **Input file:** detection_tracking.py, 02-1.avi, haarcascade_frontalface_default.xml document available in OpenCV library.
* **Output files:** output_camshift.txt, output_particle.txt, output_kalman.txt, output_of.txt.

### Implementation
Implemented face detection sng Voila Jones algorithm and tracked the face in a video using 4 different filters 
namely, Camshift filter, Particle filter, Kalman Filter and Optical Flow filter.

### Psuedo code of Kalman Filter:
```
Create Kalman Filter
Start Tracking 
While (some condition)
    x, y = track()
    Set Kalman Filter
    Change Kalman Measurements
    Predict Kalman
    Kalman Correction
    Update the center of the object
```

### How to run?
Change arguments as 
* 1 = Camshift filter 
* 2 = Particle filter
* 3 = Kalman Filter
* 4 = Optical Flow

For example, in case of Camshift filter you would run the program as follows:
```
python detection_tracking.py 1 02-1.avi ./
```
