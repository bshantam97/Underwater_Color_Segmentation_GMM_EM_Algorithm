# header files
import numpy as np
from matplotlib import pyplot as plt
import cv2

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

# initialise step of em algorithm
def initialise_step(n, d, k):
    """
    Inputs:
    n - the number of data-points
    d - dimension of gaussian
    k - number of gaussians
    
    Outputs:
    weights_gaussian - weight of gaussians of size (k)
    mean_gaussian - mean of gaussians of size (k x d)
    covariance_matrix_gaussian - covariance of gaussians of size (k x d x d)
    probability_values - probability of the datapoint being in a gaussian of size (n x k)
    """
    
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
    """
    Inputs:
    data_point - data point of the gaussian (1 x d)
    mean - mean of gaussian of size (1 x d)
    covariance - covariance of gaussian of size (1 x d x d)
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
            probability_x = probability_x + gaussian_estimation(data[j], mean_gaussian[i], covariance_matrix_gaussian[i] * weights_gaussian[i], d)
        probability_x_temp=[]    
        for i in range(k):
            val = gaussian_estimation(data[j], mean_gaussian[i], covariance_matrix_gaussian[i] * weights_gaussian[i], d) / probability_x
            probability_x_temp.append(val)
        
        # append probabilities of a point being in k-gaussians of size (1 x k)
        probabilities.append(probability_x_temp)
    return np.array(probabilities)


# update weights, maximization step
def update_weights(probabilities, k):
    """
    Inputs:
    k - number of gaussians
    probability - probability of the datapoint being in a gaussian of size (n x k)
    
    Outputs:
    updated_weights - weights of the gaussian of size (k)
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
    k - number of gaussians
    data - data to be trained on of size (n x d)
    probability_values - probability of the datapoint being in a gaussian of size (n x k)
    
    Outputs:
    updated_mean - mean of gaussians of size (k x d)
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
def update_covariance(data, probabilities, mean_gaussian, k):
    return data

# m-step of the algorithm
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def maximization_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values):
    u_weights = update_weights(probability_values, k)
    u_mean_gaussian = update_mean(data, probabilities, k)
    u_covariance_matrix_gaussian = update_covariance(data, probabilities, mean_gaussian, k)
    return (u_weights, u_mean_gaussian, u_covariance_matrix_gaussian)

# run e-m algorithm
def run_expectation_maximization_algorithm(n, d, k, iterations, data):
    """
    Inputs:
    n - the number of data-points
    d - dimension of gaussian
    k - number of gaussians
    iterations - number of iterations of the algorithm
    data - data to be trained on of size (n x d)
    
    Outputs:
    weights_gaussian - weight of gaussians of size (k)
    mean_gaussian - mean of gaussians of size (k x d)
    covariance_matrix_gaussian - covariance of gaussians of size (k x d x d)
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
