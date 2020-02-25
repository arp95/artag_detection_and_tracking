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
path_lena = ""
if(len(args) > 3):
    path_video = args[1]
    output_path_video = args[2]
    path_lena = args[3]

# define constants
dimension = 200
world_points = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]], dtype="float32")
lena_frame = cv2.imread(path_lena)
lena_frame = cv2.resize(lena_frame, (dimension - 1, dimension - 1))
cap = cv2.VideoCapture(path_video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path_video, fourcc, 20.0, (960, 540))
prev_corners = []
tracker = None

# read video
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):
        # get current video frame
        lena_list = []
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_frame, 191, 255, 0)
        
        # get corners
        lena_overlap = np.zeros(frame.shape, dtype="uint8")
        corners = get_artag_corners(frame, thresh, 0)
        for corner in corners:
            # get relation between world points and corner
            homography_matrix = get_homography_matrix(corner, world_points)
            warp_artag = warp_perspective(frame, homography_matrix, (dimension, dimension))
            gray_warp_artag = cv2.cvtColor(warp_artag, cv2.COLOR_BGR2GRAY)
            binary_code, orientation, new_world_points = get_artag_id(gray_warp_artag, dimension)
            new_homography_matrix = get_homography_matrix(new_world_points, corner)
            warp_lena = warp_perspective(lena_frame, new_homography_matrix, (frame.shape[1], frame.shape[0]))

            #val = 0
            #for index1 in range(0, warp_lena.shape[0]):
            #    for index2 in range(0, warp_lena.shape[1]):
            #        if(warp_lena[index1, index2, 0] != 0 or warp_lena[index1, index2, 1] != 0 or warp_lena[index1, index2, 2] != 0):
            #            val = val + 1

            #print(val)
            #if(val < 8000):
            lena_list.append(warp_lena)
            #lena_overlap = cv2.add(lena_overlap, warp_lena)

            if(count%50 == 0):
                print("Orientation is: " + orientation)
                print()
        
        # update black pixels in lena image
        for index1 in range(0, frame.shape[0]):
            for index2 in range(0, frame.shape[1]):
                flag = 1
                for lena in lena_list:
                    if((lena[index1, index2, 0] != 0) or (lena[index1, index2, 1] != 0) or (lena[index1, index2, 2] != 0)):
                        lena_overlap[index1, index2] = lena[index1, index2]
                        flag = 0
        
                if(flag == 1 and (lena_overlap[index1, index2, 0] == 0) and (lena_overlap[index1, index2, 1] == 0) and (lena_overlap[index1, index2, 2] == 0)):
                    lena_overlap[index1, index2] = frame[index1, index2]

        out.write(lena_overlap)
        count = count + 1
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
