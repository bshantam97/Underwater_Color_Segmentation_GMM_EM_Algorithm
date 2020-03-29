# header files
import numpy as np
from matplotlib import pyplot as plt
import cv2
from roipoly import roipoly
import pylab as pl
import sys

# set data path
args = sys.argv
if(len(args) > 2):
    path_video = args[1]
    output_path = args[2]

# define constants
cap = cv2.VideoCapture(path_video)

# read video
count = 0
while(cap.isOpened()):
    ret, frame = cap.read()    
    if(ret):
        image1 = frame.copy()

      	# draw polygon on first buoy
        pl.imshow(image1, interpolation='nearest', cmap="Greys")
        pl.colorbar()
        pl.title("left click: line segment         right click: close region")
        ROI1 = roipoly(roicolor='r')
        pl.imshow(image1, interpolation='nearest', cmap="Greys")
        pl.colorbar()
        ROI1.displayROI()
        pl.title('draw second ROI')

        # save first buoy image
        points = np.array(ROI1.get_points(image1.copy()))
        (x,y), radius = cv2.minEnclosingCircle(points)
        center = (int(x),int(y))
        radius = int(radius)
        rectX = (center[0] - radius)
        rectY = (center[1] - radius)

        cropped = image1[rectY:rectY + 2*radius, rectX:rectX + 2*radius].copy()
        cv2.imwrite(str(output_path) + "/buoy" + str(count) + ".png", cropped)
        count = count + 1
    else:
        break
cap.release()
cv2.destroyAllWindows()
