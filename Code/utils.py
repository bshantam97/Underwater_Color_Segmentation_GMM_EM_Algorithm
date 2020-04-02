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
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy


# initialise step of em algorithm
def initialise_step(n, d, k):
    """
    Inputs:
    n - number of datapoints
    d - dimension of the gaussian
    k - number of the gaussians
    
    Outputs:
    weights_gaussian - weight of the gaussians, size (k)
    mean_gaussian - mean of the gaussians, size (k x d)
    covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    probability_values - probability of the datapoint being in the k-gaussians, size (n x k)
    """
    
    # initialise weights
    weights_gaussian = np.zeros(k)
    for index in range(0, k):
        weights_gaussian[index] = (1.0 / k)
    
    # initialise mean
    mean_gaussian = np.zeros((k, d))
    
    # initialise covariance
    covariance_matrix_gaussian = np.zeros((k, d, d))
    
    # randomly initialise probability
    probability_values = np.zeros((n, k))
    for index in range(0, n):
        probability_values[index][np.random.randint(0, k)] = 1
        
    # return the arrays
    return (weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)


# gaussian estimation for expectation step
def gaussian_estimation(data_point, mean, covariance, dimension):
    """
    Inputs:
    data_point - data point of the gaussian, size (1 x d)
    mean - mean of the gaussian, size (1 x d)
    covariance - covariance of the gaussian, size (1 x d x d)
    dimension - dimension of the gaussian
    
    Outputs:
    value of the gaussian
    """
    
    determinant_covariance = np.linalg.det(covariance)
    determinant_covariance_root = np.sqrt(determinant_covariance)
    covariance_inverse = np.linalg.inv(covariance)
    gaussian_pi_coeff = 1.0 / np.power((2 * np.pi), (dimension / 2))
    data_mean_diff = (data_point - mean)
    data_mean_diff_transpose = data_mean_diff.T     
    return (gaussian_pi_coeff) * (determinant_covariance_root) * np.exp(-0.5 * np.matmul(np.matmul(data_mean_diff, covariance_inverse), data_mean_diff_transpose))


# gaussian estimation for n-points
def gaussian_estimation_array(data_point, mean, covariance, dimension):
    """
    Inputs:
    data_point - data point of the gaussian, size (n x d)
    mean - mean of the gaussian, size (1 x d)
    covariance - covariance of the gaussian, size (1 x d x d)
    dimension - dimension of the gaussian
    
    Outputs:
    value of the gaussian, size (n x d)
    """
    
    determinant_covariance = np.linalg.det(covariance)
    determinant_covariance_root = np.sqrt(determinant_covariance)
    covariance_inverse = np.linalg.inv(covariance)
    gaussian_pi_coeff = 1.0 / np.power((2 * np.pi), (dimension / 2))
    data_mean_diff = (data_point - mean)
    data_mean_diff_transpose = data_mean_diff.T 
    val = (gaussian_pi_coeff) * (determinant_covariance_root) * np.exp(-0.5 * np.sum(np.multiply(data_mean_diff * covariance_inverse, data_mean_diff), axis=1))
    return np.reshape(val, (data_point.shape[0], data_point.shape[1]))


# gaussian estimation for 3-dimensional case
def gaussian_estimation_3d(data_point, mean, cov):
    det_cov = np.linalg.det(cov)
    cov_inv = np.zeros_like(cov)
    mean = np.array(mean)
    cov = np.array(cov)
    for i in range(data_point.shape[1]):
        cov_inv[i, i] = 1 / cov[i, i]
    diff = np.matrix(data_point - mean)
    return (2.0 * np.pi) ** (-len(data_point[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(diff * cov_inv, diff), axis=1))


# e-step of the algorithm
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def expectation_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values):
    """
    Inputs:
    n - the number of data-points
    d - dimension of gaussian
    k - number of gaussians
    data - data to be trained on of size (n x d)
    weights_gaussian - weight of gaussians of size (k)
    mean_gaussian - mean of gaussians of size (k x d)
    covariance_matrix_gaussian - covariance of gaussians of size (k x d x d)
    probability_values - probability of the datapoint being in a gaussian of size (n x k)
    
    Outputs:
    probabilities - probability array of size (n x k)
    """
    
    # create empty array of list of probabilities
    probabilities = []
    
    # iterate through each item
    for j in range(0, n):
        
        # calculate probability of a point being in the k-gaussians
        probability_x = 0.0
        for i in range(0, k):
            probability_x = probability_x + gaussian_estimation(data[j], mean_gaussian[i], covariance_matrix_gaussian[i], d) * weights_gaussian[i]
        probability_x_temp = []    
        for i in range(0, k):
            val = (gaussian_estimation(data[j], mean_gaussian[i], covariance_matrix_gaussian[i], d) * weights_gaussian[i]) / probability_x
            probability_x_temp.append(val)
        
        # append probabilities of a point being in k-gaussians of size (1 x k)
        probabilities.append(probability_x_temp)
    return np.array(probabilities)


# update weights, maximization step
def update_weights(probabilities, k):
    """
    Inputs:
    probabilities - probability of the datapoint being in the k-gaussians, size (n x k)
    k - number of gaussians
    
    Outputs:
    updated_weights - weights of the k-gaussians, size (k)
    """
    
    probabilities = np.array(probabilities)
    updated_weights = []
    for i in range(0, k):
        updated_weights.append(np.sum(probabilities[:, i]))
    updated_weights = np.array(updated_weights)
    return updated_weights / np.sum(updated_weights)


# update mean, maximization step
def update_mean(data, probabilities, k):
    """
    Inputs:
    data - training data, size (n x d)
    probabilities - probability of the datapoints being in k-gaussians, size (n x k)
    k - number of the gaussians
    
    Outputs:
    updated_mean - mean of the k-gaussians, size (k x d)
    """
    
    probabilities = np.array(probabilities)
    data = np.array(data)
    updated_weights = []
    updated_mean = np.matmul(probabilities.T, data)
    for i in range(0, k):
        updated_weights.append(np.sum(probabilities[:, i]))
        updated_mean[i] = updated_mean[i] / updated_weights[i]
    return updated_mean


# update covariance, maximization step
def update_covariance(data, probabilities_values, mean_gaussian, k, d, n):
    """
    Inputs:
    data - training data, size (n x d)
    probability_values - probability of the datapoint being in k-gaussians, size (n x k)
    mean_gaussian - mean of the k-gaussians, size (k x d)
    k - number of the gaussians
    d - dimension of the gaussian
    n - number of data-points
    
    Outputs:
    k_array - probability array, size (n x k)
    """
    
    probabilities_values = np.array(probabilities_values)
    mean_gaussian = np.array(mean_gaussian)
    data = np.array(data)
    probabilities_sum = []
    k_array = []
    for i in range(0, k):
        probabilities_sum.append(np.sum(probabilities_values[:, i]))
        covariance_array = []
        for index1 in range(0, d):
            temp_array = []
            for index2 in range(0, d):
                check = 0
                for index3 in range(0, n):
                    check = check + (probabilities_values[index3, i] * (data[index3, index1] - mean_gaussian[i, index1]) * (data[index3, index2] - mean_gaussian[i, index2]))
                check = check / probabilities_sum[i]
                if(index1 == index2):
                    if(np.abs(check) < 0.0001):
                        check = 0.0001
                temp_array.append(check)
            covariance_array.append(temp_array)
        k_array.append(covariance_array)
    return k_array


# m-step of the algorithm
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def maximization_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values):
    """
    Inputs:
    n - number of data-points
    d - dimension of gaussian
    k - number of gaussians
    data - training data, size (n x d)
    weights_gaussian - weight of the gaussians, size (k)
    mean_gaussian - mean of the gaussians, size (k x d)
    covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    probability_values - probability of the datapoint being in a gaussian, size (n x k)
    
    Outputs:
    u_weights - weight of the gaussians, size (k)
    u_mean_gaussian - mean of the gaussians, size (k x d)
    u_covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    """
    u_weights = update_weights(probability_values, k)
    u_mean_gaussian = update_mean(data, probability_values, k)
    u_covariance_matrix_gaussian = update_covariance(data, probability_values, mean_gaussian, k, d, n)
    return (u_weights, u_mean_gaussian, u_covariance_matrix_gaussian)


# run e-m algorithm
def run_expectation_maximization_algorithm(n, d, k, iterations, data):
    """
    Inputs:
    n - number of data-points
    d - dimension of gaussian
    k - number of gaussians
    iterations - number of iterations 
    data - training data, size (n x d)
    
    Outputs:
    weights_gaussian - weight of the gaussians, size (k)
    mean_gaussian - mean of the gaussians, size (k x d)
    covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    """
    
    # initialise step
    (weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values) = initialise_step(n, d, k)
    
    # run for fixed iterations
    for i in range(0, iterations):
    
        # m-step
        (weights_gaussian, mean_gaussian, covariance_matrix_gaussian) = maximization_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)
    
        # e-step
        probability_values = expectation_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)
    
    # return answer
    return (weights_gaussian, mean_gaussian, covariance_matrix_gaussian)


# plot histogram
def plot_hist(image):
    # loop over the image channels
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    features = []
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        features.extend(hist)
        plt.plot(hist, color = color)
        plt.xlim([0, 256])


# data for training
def get_training_data(file_path, channel1, channel2, channel3):
    data = []
    files = glob.glob(file_path + "/*")
    for file in files:
        image = cv2.imread(file)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                val = []
                if(channel1):
                    val.append(image[row, col, 0])
                if(channel2):
                    val.append(image[row, col, 1])
                if(channel3):
                    val.append(image[row, col, 2])
                data.append(val)
    return np.array(data)


# reference: https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
        
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)
