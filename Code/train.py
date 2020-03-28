# header files
import numpy as np
from matplotlib import pyplot as plt
import cv2
from roipoly import roipoly
import pylab as pl

# set data path
path_video = "/mnt/c/Users/shant/Underwater_Color_Segmentation_GMM_EM_Algorithm/data/detectbuoy.avi"

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

        cv2.imwrite("/mnt/c/Users/shant/Underwater_Color_Segmentation_GMM_EM_Algorithm/data/buoy2/buoy" + str(count) + ".png", cropped)
        
        #cv2.imshow("Image", cropped)
        #cv2.waitKey(0)

        # draw polygon on second buoy
        '''
        pl.imshow(image2.copy(), interpolation='nearest', cmap="Greys")
        pl.colorbar()
        pl.title("left click: line segment         right click: close region")
        ROI2 = roipoly(roicolor='r')
        pl.imshow(image2.copy(), interpolation='nearest', cmap="Greys")
        pl.colorbar()
        ROI2.displayROI()
        pl.title('draw second ROI')

        # save second buoy image
        points = np.array(ROI2.get_points(image2.copy()))
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped = image2[y:y+h, x:x+w].copy()
        cv2.imwrite("data/buoy2/buoy" + str(count) + ".png", cropped)
        cv2.imshow("Image", cropped)
        cv2.waitKey(0)

        # draw polygon on third buoy
        pl.imshow(image3.copy(), interpolation='nearest', cmap="Greys")
        pl.colorbar()
        pl.title("left click: line segment         right click: close region")
        ROI3 = roipoly(roicolor='r')
        pl.imshow(image3.copy(), interpolation='nearest', cmap="Greys")
        pl.colorbar()
        ROI3.displayROI()
        pl.title('draw second ROI')

        # save third buoy image
        points = np.array(ROI3.get_points(image3.copy()))
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped = image3[y:y+h, x:x+w].copy()
        cv2.imwrite("data/buoy3/buoy" + str(count) + ".png", cropped)
        cv2.imshow("Image", cropped)
        cv2.waitKey(0)
        break
        '''
        count = count + 1
    else:
        break
cap.release()
cv2.destroyAllWindows()
