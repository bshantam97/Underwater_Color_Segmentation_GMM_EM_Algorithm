# header files
import numpy as np
from matplotlib import pyplot as plt
import cv2

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

# initialise step of em algorithm
def initialise_step(n, d, k):
    weights_gaussian = np.zeros(k)
    mean_gaussian = np.zeros((k, d))
    covariance_matrix_gaussian = np.zeros((k, d, d))
    probability_values = np.zeros((n, k))
    
    # randomly assign probability values
    for index in range(0, n):
        probability_values[index][np.random.randint(0, k)] = 1
        
    # return the arrays
    return (weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)
    
# gaussian estimation for expectation step
def gaussian_estimation(data_point, mean, covariance, dimension):
    determinant_covariance = np.linalg.det(covariance)
    determinant_covariance_root = np.sqrt(determinant_covariance)
    covariance_inverse = np.linalg.inv(covariance)
    gaussian_pi_coeff = 1 / np.power((2 * np.pi), (dimension / 2))
    data_mean_diff = (data_point - mean)
    data_mean_diff_transpose = data_mean_diff.T     
    return (gaussian_pi_coeff) * (determinant_covariance_root) * np.exp(-0.5 * np.matmul(np.matmul(data_mean_diff_transpose, covariance_inverse), data_mean_diff))

# e-step of the algorithm
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def expectation_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values):
    probabilities = []
    for j in range(0, n):
        probability_x = 0.0
        for i in range(0, k):
            probability_x = probability_x + gaussian_estimation(data.iloc[j], mean_gaussian[i], covariance_matrix_gaussian[i] * weights_gaussian[i], d)
        probability_x_temp=[]    
        for i in range(k):
            val = gaussian_estimation(data.iloc[j], mean_gaussian[i], covariance_matrix_gaussian[i] * weights_gaussian[i], d) / probability_x
            probability_x_temp.append(val)
        probabilities.append(probability_x_temp)
    return pd.DataFrame(probabilities)
