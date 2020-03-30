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
