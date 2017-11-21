# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_color = cv2.resize(cv2.imread("images/pattern001.jpg"), (0,0), fx=scale_factor,fy=scale_factor)
    
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        
		# Populate scan_bits by putting the bit_code according to on_mask
        for true_bit in zip(*np.where(on_mask)):
            scan_bits[true_bit[0], true_bit[1]] |= bit_code
			
    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    camera_rgb_values = []
    corr_img = np.zeros((h, w, 3), dtype=np.float64)
    norm_img = np.zeros((h, w, 3), dtype=np.float64)
	
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code

            # Use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # find for the camera (x,y) the projector (x_p, y_p).
            # store your points in camera_points and projector_points
			# IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
			
			# Obtain x_p, y_p - the projector coordinates from the codebook
            x_p , y_p = binary_codes_ids_codebook[scan_bits[y,x]]
			
			# Adding filter to reduce errors decoding the patterns
            if x_p >= 1279 or y_p >= 799:  #filter
                continue
			
            projector_points.append([[x_p, y_p]])
            camera_points.append([[x/2.0, y/2.0]])
            camera_rgb_values.append(ref_color[y, x].tolist())
			
            corr_img[y][x] = [x_p, y_p, 0]
	
    # RGB format in MatplotLib and BGR format in openCV
    r = corr_img[:,:,0]
    g = corr_img[:,:,1]
    b = corr_img[:,:,2]
	
    norm_img[:,:,0]=(r - r.min())/(r.max() - r.min())*255.0
    norm_img[:,:,1]=(g - g.min())/(g.max() - g.min())*255.0   
    norm_img=cv2.convertScaleAbs(norm_img)
	
    # The image to show correspondences between camera and projector 
    plt.imshow(norm_img)
    plt.imsave('correspondence.jpg',norm_img)
    plt.show()
			
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

        # Use cv2.undistortPoints to get normalized points. For camera, use camera_K and camera_d
        # Use cv2.undistortPoints to get normalized points. For projector, use projector_K and projector_d
        camera_norm = cv2.undistortPoints(np.array(camera_points, dtype=np.float64), camera_K, camera_d)
        projector_norm = cv2.undistortPoints(np.array(projector_points, dtype=np.float64), projector_K, projector_d)

        # Use cv2.triangulatePoints to triangulate the normalized points
        proj_matrix_1 = np.hstack((projector_R, projector_t))
        proj_matrix_2 = np.hstack((np.eye(3), np.zeros((3,1)) ))
        tri_points = cv2.triangulatePoints(proj_matrix_1, proj_matrix_2, projector_norm, camera_norm)
		
        # Use cv2.convertPointsFromHomogeneous to get real 3D points
        # name the resulted 3D points as "points_3d"
        points_3d = cv2.convertPointsFromHomogeneous(tri_points.T)

        camera_rgb_values = np.array(camera_rgb_values)
        points_3d_color = np.zeros((points_3d.shape[0], points_3d.shape[1], 6), dtype=np.float64)
        for i in range(0, points_3d.shape[0]):
            points_3d_color[i][0] =  np.append(points_3d[i][0], camera_rgb_values[i])
			
		# apply another filter on the Z-component
        mask = (points_3d_color[:,:,2] > 200) & (points_3d_color[:,:,2] < 1400)
        points_3d_color = points_3d_color[mask]
	
	return points_3d_color
	
def write_3d_points(points_3d):
	
	# ===== DO NOT CHANGE THIS FUNCTION =====
	
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    output_name_color = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0],p[1],p[2]))
			
    with open(output_name_color,"w") as f:
        for p in points_3d:
            f.write("%d %d %d %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))

    return points_3d
    
if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)