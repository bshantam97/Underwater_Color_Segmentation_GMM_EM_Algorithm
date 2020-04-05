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


# set data path, dimension of gaussian, number of gaussians and number of iterations
k = 2
d = 1
iterations = 30
args = sys.argv
video_path = ""
file_path = ""
if(len(args) > 2):
    video_path = args[1]
    file_path = args[2]

# get training data
training_data1 = get_training_data(file_path, 0, 0, 1)
training_data1 = training_data1[:10000, :]
training_data2 = get_training_data(file_path, 0, 1, 0)
training_data2 = training_data2[:10000, :]
training_data = np.vstack((training_data1, training_data2))

# get the weights, mean and variances of gaussian
(weights_gaussian, mean_gaussian, covariance_matrix_gaussian) = run_expectation_maximization_algorithm(training_data.shape[0], d, k, iterations, training_data)

print(weights_gaussian)
print(mean_gaussian)
print(covariance_matrix_gaussian)

# segmenting the buoy
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('yellow_buoy_1D_gauss.avi', fourcc, 5.0, (640, 480))
while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break    

    # steps to find the probability of each pixel in the k-gaussians
    image1 = frame[:, :, 1].ravel()
    image2 = frame[:, :, 2].ravel()
    image = np.concatenate((image1, image2), axis=0)
    image = np.reshape(image, (image.shape[0], d))
    prob = np.zeros((image.shape[0], k))
    likelihood = np.zeros((image.shape[0], k))
    for index in range(0, k):
        prob[: ,index:index+1] = gaussian_estimation_array(image, mean_gaussian[index], covariance_matrix_gaussian[index], 1) * weights_gaussian[index]
        likelihood = prob.sum(1)
    
    # pre-process image and create a binary image
    prob_green = likelihood[: frame.shape[0] * frame.shape[1]]
    prob_red = likelihood[frame.shape[0] * frame.shape[1]:]
    prob = np.add(prob_red, prob_green)
    prob[prob_red > np.max(prob_red) / 3.0] = 255
    output_image = np.zeros_like(frame)
    output_image[:, :, 1] = np.reshape(prob, (frame.shape[0], frame.shape[1]))
    output_image[:, :, 2] = np.reshape(prob, (frame.shape[0], frame.shape[1]))
    gray_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.GaussianBlur(gray_output_image, (5, 5), 2)
    _, edged = cv2.threshold(output_image, 30, 255, 0)
    
    # find contours and segment the orange buoy
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts, _ = sort_contours(cnts, method="right-to-left")
    hull = cv2.convexHull(cnts[0])
    (x, y), radius = cv2.minEnclosingCircle(hull)
    if radius > 6:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        out.write(frame)
    else:
        out.write(frame)
