"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal Shantam Bajpai
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""

# header files
from utils import *
import sys

# set data path
args = sys.argv
path_video = ""
output_path_video = ""
if(len(args) > 1):
    path_video = args[1]

# define constants
dimension = 200
points_3d = np.float32([[0, 0, 0], [0, 200, 0], [200, 200, 0], [200, 0, 0], [0, 0, -200], 
                               [0, 200, -200], [200, 200, -200], [200, 0, -200]])
world_points = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]], dtype="float32")
camera_matrix = np.array([[1406.08415449821, 0, 0], 
                                      [2.20679787308599, 1417.99930662800, 0], 
                                      [1014.13643417416, 566.347754321696, 1]]).T
cap = cv2.VideoCapture(path_video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_problem2b.avi", fourcc, 20.0, (960, 540))

# read video
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        # get current video frame
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_frame, 191, 255, 0)
        
        # get corners
        frame_black = np.zeros(frame.shape, dtype='uint8')
        corners = get_artag_corners(frame, thresh, 0)

        for corner in corners:
            # get relation between world points and corner
            homography_matrix = get_homography_matrix(corner, world_points)
            warp_artag = warp_perspective(frame, homography_matrix, (dimension, dimension))
            gray_warp_artag = cv2.cvtColor(warp_artag, cv2.COLOR_BGR2GRAY)
            binary_code, orientation, new_world_points = get_artag_id(gray_warp_artag, dimension)
            new_homography_matrix = get_homography_matrix(new_world_points, corner)
            (projection_matrix, rotation_matrix, translation_matrix) = get_projection_matrix(new_homography_matrix, camera_matrix)
            points = project_points(points_3d, camera_matrix, rotation_matrix, translation_matrix)
            cube_image = draw_cube_on_image(frame_black.copy(), points)
            frame_black = cv2.add(frame_black, cube_image)

            if(count%50 == 0):
                print("Orientation is: " + orientation)
                print()
        
        # update black pixels in lena image
        for index1 in range(0, frame.shape[0]):
            for index2 in range(0, frame.shape[1]):
                if((frame_black[index1, index2, 0] == 0) and (frame_black[index1, index2, 1] == 0) and (frame_black[index1, index2, 2] == 0)): 
                    frame_black[index1, index2] = frame[index1, index2]

        out.write(frame_black)
        count = count + 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
