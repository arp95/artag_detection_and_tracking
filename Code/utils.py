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
import cv2
import numpy as np
from scipy.spatial import distance as dist


# Get ID of the artag (using the birds-eye view image of the artag, obtained from the homography matrix)
def get_artag_id(image, dimension):
    ret, image_thresh = cv2.threshold(image, 190, 255, 0)
    cropped_image_thresh = image_thresh[50:150, 50:150]
    
    # get id of artag
    first_bit = 0
    second_bit = 0
    third_bit = 0
    fourth_bit = 0
    for index1 in range(30, 40):
        for index2 in range(30, 40):
            if(cropped_image_thresh[index1, index2] >= 250):
                first_bit = 1
                break
    
    for index1 in range(30, 40):
        for index2 in range(60, 70):
            if(cropped_image_thresh[index1, index2] >= 250):
                second_bit = 1
                break
    
    for index1 in range(60, 70):
        for index2 in range(60, 70):
            if(cropped_image_thresh[index1, index2] >= 250):
                third_bit = 1
                break
                
    for index1 in range(60, 70):
        for index2 in range(30, 40):
            if(cropped_image_thresh[index1, index2] >= 250):
                fourth_bit = 1
                break
    
    # get orientation for the artag
    top_left = 0
    bottom_left = 0
    top_right = 0
    bottom_right = 0
    for index1 in range(2, 10):
        for index2 in range(2, 10):
            if(cropped_image_thresh[index1, index2] >= 250):
                top_left = 1
                break
       
    if(top_left == 0):
        for index1 in range(90, 98):
            for index2 in range(2, 10):
                if(cropped_image_thresh[index1, index2] >= 250):
                    bottom_left = 1
                    break
    
    if(top_left == 0 and bottom_left == 0):
        for index1 in range(2, 10):
            for index2 in range(90, 98):
                if(cropped_image_thresh[index1, index2] >= 250):
                    top_right = 1
                    break
                    
    if(top_left == 0 and bottom_left == 0 and top_right == 0):
        for index1 in range(90, 98):
            for index2 in range(90, 98):
                if(cropped_image_thresh[index1, index2] >= 250):
                    bottom_right = 1
                    break
    
    # get orientation of ar tag
    if(top_right == 1):
        new_points = np.array([[dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1], [0, 0]], dtype="float32")
        return (np.array([third_bit, second_bit, first_bit, fourth_bit]), "TR", new_points)
    elif(top_left == 1):
        new_points = np.array([[dimension - 1, dimension - 1], [0, dimension - 1], [0, 0], [dimension - 1, 0]], dtype="float32")
        return (np.array([second_bit, first_bit, fourth_bit, third_bit]), "TL", new_points)
    elif(bottom_left == 1):
        new_points = np.array([[0, dimension - 1], [0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1]], dtype="float32")
        return (np.array([first_bit, fourth_bit, third_bit, second_bit]), "BL", new_points)
    else:
        new_points = np.array([[0, 0], [dimension - 1, 0], [dimension - 1, dimension - 1], [0, dimension - 1]], dtype="float32")
        return (np.array([fourth_bit, third_bit, second_bit, first_bit]), "BR", new_points)

    
# function to order points from top-left to bottom-right (used for calculation of homography matrix)
# reference: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def order_points(points):
    # sort points along columns
    col_sorted_points = points[np.argsort(points[:, 0]), :]
    
    # left_col_points have top left and bottom left and right_col_points have top right and bottom right
    left_col_points = col_sorted_points[:2, :]
    right_col_points = col_sorted_points[2:, :]
    
    # sort leftmost according to rows
    left_col_points = left_col_points[np.argsort(left_col_points[:, 1]), :]
    (tl, bl) = left_col_points
    
    # now get top right and bottom right
    euc_dist = dist.cdist(tl[np.newaxis], right_col_points, "euclidean")[0]
    (br, tr) = right_col_points[np.argsort(euc_dist)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

    
# find corners(tl, tr, br, bl) of the ar-tags
def get_artag_corners(frame, thresh, flag):
    # find contours and order them for finding homography matrix
    arTag_corners = []
    (contours, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    
    # loop through each contour found and keep only the relevant
    for component in zip(contours, hierarchy):
        currContour = component[0]
        currHierarchy = component[1]
        peri = cv2.arcLength(currContour, True)
        approx = cv2.approxPolyDP(currContour, 0.015*peri, True)
        start_col, start_row, width, height = cv2.boundingRect(currContour)
        
        if(currHierarchy[3] != -1 and len(approx) == 4 and  cv2.contourArea(approx) < 8000 and cv2.contourArea(approx) > 100):
            if(flag):
                cv2.drawContours(frame, [currContour], 0, (0, 255, 0), 3)
            corners = np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            corners = order_points(corners)
            arTag_corners.append(corners)
    return np.array(arTag_corners)


# function equivalent to cv2.warpPerspective
def warp_perspective(image, homography_matrix, dimension):
    # find the birds-eye view of image using homography matrix passed as input
    image = cv2.transpose(image)
    warped_image = np.zeros((dimension[0], dimension[1], 3))
    for index1 in range(0, image.shape[0]):
        for index2 in range(0, image.shape[1]):
            new_vec = np.dot(homography_matrix, [index1, index2, 1])
            new_row, new_col, _ = (new_vec / new_vec[2] + 0.4).astype(int)
            if new_row > 5 and new_row < (dimension[0] - 5):
                if new_col > 5 and new_col < (dimension[1] - 5):
                    warped_image[new_row, new_col] = image[index1, index2]
                    warped_image[new_row-1, new_col-1] = image[index1, index2]
                    warped_image[new_row-2, new_col-2] = image[index1, index2]
                    warped_image[new_row-3, new_col-3] = image[index1, index2]
                    warped_image[new_row+1, new_col+1] = image[index1, index2]
                    warped_image[new_row+2, new_col+2] = image[index1, index2]
                    warped_image[new_row+3, new_col+3] = image[index1, index2]
     
    # convert matrix to image
    warped_image = np.array(warped_image, dtype=np.uint8)
    warped_image = cv2.transpose(warped_image)
    return warped_image


# find homography matrix using the four point correspondences
# reference: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
# reference: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
def get_homography_matrix(corner, world_points):
    # construct a matrix
    a_matrix = []
    for index in range(0, len(corner)):
        x, y = corner[index][0], corner[index][1]
        u, v = world_points[index][0], world_points[index][1]
        a_matrix.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a_matrix.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    
    # compute A.A(T) and A(T). A
    a_matrix = np.array(a_matrix)
    eig_values_1, eig_vects_1 = np.linalg.eig(np.matmul(a_matrix, a_matrix.T))
    eig_values_2, eig_vects_2 = np.linalg.eig(np.matmul(a_matrix.T, a_matrix))

    # sort eigenvalues and get the corresponding eigenvectors
    index_1 = eig_values_1.argsort()[::-1]
    eig_values_1 = eig_values_1[index_1]
    eig_vects_1 = eig_vects_1[:, index_1]
    index_2 = eig_values_2.argsort()[::-1]
    eig_values_2 = eig_values_2[index_2]
    eig_vects_2 = eig_vects_2[:, index_2]

    # compute homography matrix
    v_matrix = eig_vects_2
    homography_mat = np.zeros((a_matrix.shape[1], 1))
    for index in range(0, a_matrix.shape[1]):
        homography_mat[index, 0] = v_matrix[index, v_matrix.shape[1] - 1]
    homography_mat = homography_mat.reshape((3, 3))
    
    # scale the homography matrix by h[3][3] element
    for index1 in range(0, 3):
        for index2 in range(0, 3):
            homography_mat[index1][index2] = homography_mat[index1][index2] / homography_mat[2][2]
    return homography_mat


# find projection matrix using homography matrix and camera matrix
def get_projection_matrix(homography_matrix, camera_matrix):
    # calculate bhat matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    bhat_matrix = np.dot(camera_matrix_inv, homography_matrix)
    
    # calculate lambda constant
    lambda_constant = ((np.linalg.norm(np.dot(camera_matrix_inv, homography_matrix[:, 0])) + np.linalg.norm(np.dot(camera_matrix_inv, homography_matrix[:, 1])))/2)
    lambda_constant = 1.0/lambda_constant
    
    # calculate b matrix
    bhat_det = np.linalg.det(bhat_matrix)
    if(bhat_det < 0):
        b_matrix = (-bhat_matrix)*(lambda_constant)
    else:
        b_matrix = (bhat_matrix)*(lambda_constant)
        
    # calculate rotation and translation matrices
    rotation_matrix_1 = (b_matrix[:, 0])*(lambda_constant)
    rotation_matrix_2 = (b_matrix[:, 1])*(lambda_constant)
    rotation_matrix_3 = np.cross(rotation_matrix_1, rotation_matrix_2)/(lambda_constant)
    translation_matrix = np.array([b_matrix[:, 2]*lambda_constant]).T
    rotation_matrix = np.array([rotation_matrix_1, rotation_matrix_2, rotation_matrix_3]).T
    net_matrix = np.hstack([rotation_matrix, translation_matrix])
    
    # calculate projection matrix
    projection_matrix = np.dot(camera_matrix, net_matrix)
    return (projection_matrix, rotation_matrix, translation_matrix)


# uses 2d points from project_points function to draw cube on image
# reference: https://www.learnopencv.com/tag/projection-matrix/
def draw_cube_on_image(image, points):
    # ground part
    image = cv2.drawContours(image, [points[:4]], -1, (0, 255, 0), 3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        image = cv2.line(image, tuple(points[i]), tuple(points[j]), (255, 0, 0), 3)

    # top part
    image = cv2.drawContours(image, [points[4:]], -1, (0, 0, 255), 3)
    return image


# project 3d points to 2d points
# reference: https://stackoverflow.com/questions/28180413/why-is-cv2-projectpoints-not-behaving-as-i-expect
def project_points(points_3d, camera_matrix, rotation_matrix, translation_matrix):
    points_2d = []
    for point in points_3d:
        point = np.matmul(np.matmul(camera_matrix, rotation_matrix), np.array([point]).T) + np.matmul(camera_matrix, translation_matrix)
        point = point/point[2]
        points_2d.append([int(point[0]), int(point[1])])
    return np.array(points_2d)
