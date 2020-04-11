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
import dlib

# set data path
args = sys.argv
path_video = ""
output_path_video = ""
if(len(args) > 1):
    path_video = args[1]

# define constants
dimension = 200
world_points = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]], dtype="float32")
cap = cv2.VideoCapture(path_video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_problem1.avi", fourcc, 20.0, (960, 540))
tracker = None

# read video
count = 0
prev_corners = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        # get current video frame
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_frame, 191, 255, 0)
        
        # get corners
        corners = get_artag_corners(frame, thresh, 1)
        for corner in corners:
            # get relation between world points and corner
            homography_matrix = get_homography_matrix(corner, world_points)
            warp_artag = warp_perspective(frame, homography_matrix, (dimension, dimension))
            gray_warp_artag = cv2.cvtColor(warp_artag, cv2.COLOR_BGR2GRAY)
            binary_code, orientation, new_world_points = get_artag_id(gray_warp_artag, dimension)

            if(count%20 == 0):
                print("Orientation is: " + orientation)
                print("ID is: " + str(binary_code))
                print()

        out.write(frame)
        count = count + 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
