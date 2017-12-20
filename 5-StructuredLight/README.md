 
## Your goal is to reconstruct a scene from multiple structured light scannings of it.

* Calibrate projector with the “easy” method
* Use ray-plane intersection to get 2D-3D correspondence and use stereo calibration
* Get the binary code for each pixel - this you should do, but it's super easy
* Correlate code with (x,y) position - we provide a "codebook" from binary code -> (x,y)
* With 2D-2D correspondence, perform stereo triangulation (existing function) to get a depth map

### Run the program as 
```
 python reconstruct.py ./
```
### Input given:
* Stereo calibration
  * K and distortion for projector and camera
  * R, and t between projector and camera
* Images of binary codes pattern for depth scanning

### Output:
* correspondence.jpg : Shows the correspondences between camera and projector.
* output.xyz : Final 3d points. Can be rendered using CloudCompare. http://www.danielgm.net/cc/release/

**NOTE:** The colored output.xyz is wrong. Remember OpenCV uses BGR order.
