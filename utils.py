import cv2
import numpy as np
from skimage import measure

def findBoundaryBoxes(srcImage, minArea=3000, contourThresh=2.0):
	hist = np.histogram(srcImage, bins=np.arange(0, 256))
	indexMaxValue = np.argmax(hist[0])
	threshValue = indexMaxValue + 30
	
	_, threshImg = cv2.threshold(srcImage, threshValue, 255, cv2.THRESH_BINARY)

	contours = measure.find_contours(threshImg, contourThresh)

	retValues = []
	for contour in contours:
		x = contour[:, 1]
		y = contour[:, 0]
		xmin = int(min(x)) - 10
		ymin = int(min(y)) - 10
		xmax = int(max(x)) + 10
		ymax = int(max(y)) + 10
		area = (xmax-xmin)*(ymax-ymin)
		
		if area>minArea:
			retValues.append((xmin, ymin, xmax, ymax))

	return retValues




if __name__ == '__main__':
	print('Utils Routines!')