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
def initialise(n, d, k):
    weights_gaussian = np.zeros(k)
    mean_gaussian = np.zeros(k, d)
    covariance_matrix_gaussian = np.zeros(k, d, d)
    probability_values = np.zeros(n, k)
    
    # randomly assign probability values
    for index in range(0, n):
        probability_values[index][np.random.randint(0, k)] = 1
        
    # return the arrays
    return (weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)
    
# gaussian estimation for expectation step
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def gaussian_estimation(data_point, mean, sig, d):
    data_point_mean = data_point - mean
    sig_inv = np.linalg.inv(sig)
    in_exp = np.exp(-1 * 0.5 * np.matmul(np.matmul(data_point_mean, sig_inv), data_point_mean.T))
    a = np.power(2 * np.pi, d) * np.linalg.det(sig)
    out_exp= 1 / np.sqrt(a)
    return (in_exp * out_exp)

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
