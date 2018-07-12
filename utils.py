import sys
import numpy as np
import cv2
from skimage import measure
from PIL import Image, ImageGrab
import warnings
warnings.filterwarnings("ignore")

#Set the EIGENVEC_REDUCTION factor. Must be the same that was used in "/SklearnClassifier(Number-Recognition)/NumberPredict/sklearn_Train.py
EIGENVEC_REDUCTION = 1.4

#Set the shape of Samples used to train the NumberClassifier in "/SklearnClassifier(Number-Recognition)/NumberPredict/sklearn_Train.py
TRAIN_WIDTH = 50
TRAIN_HEIGHT = 65

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

def enhanceDigit(digitImage, thrOffset=50, scaleFactor=0.2):
	hist, _ = np.histogram(digitImage, 256, [0,256])
	indexMaxValue = np.argmax(hist)
	binaryThreshold = indexMaxValue + thrOffset
	_ ,thrDigit = cv2.threshold(digitImage, binaryThreshold, 255, cv2.THRESH_BINARY_INV)

	if scaleFactor > 0:
		h, w = thrDigit.shape
		thrDigit = cv2.resize(thrDigit, (0,0), fx=scaleFactor, fy=scaleFactor)

	return thrDigit

def getRawScreenshot(grayImg=False, PILFormat=False, bbox=None, cv2ResizeBbox=None):
	try:
		screenCap = ImageGrab.grab()
	except Exception as e:
		print(str(e))
	screenCap = np.array(screenCap) 
	
	if grayImg:
		screenCap = cv2.cvtColor(screenCap, cv2.COLOR_RGB2GRAY)
	else:
		screenCap = cv2.cvtColor(screenCap, cv2.COLOR_RGB2BGR)
	
	if isinstance(bbox, tuple):
		screenCap = screenCap[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
		if isinstance(cv2ResizeBbox, tuple):
			screenCap = cv2.resize(screenCap, cv2ResizeBbox)

	if PILFormat:
		screenCap = cv2.cvtColor(screenCap, cv2.COLOR_BGR2RGB)
		return Image.fromarray(screenCap)
	else:
		return screenCap

def recognizeDigit(digit, numberClassifier):
	#Check Digit shape
	w, h = digit.shape
	if w<=0 or h<=0:
		return -1
	#Shape the image in order to fit in Number Classifier
	digit = cv2.resize(digit, (TRAIN_WIDTH, TRAIN_HEIGHT))
	#Enhace the image
	digit = enhanceDigit(digit)
	
	#EXTRACT FEATURES
	#Mean normalize all rows
	digit = digit.astype(np.float32)
	for row in range(digit.shape[0]):
		u = digit[row, :].mean()
		digit[row, :] = digit[row, :] - u

	#Extract main eigenvectors
	_, _, Vh = np.linalg.svd(digit)
	p_eigVec = Vh[0:int(len(Vh)/EIGENVEC_REDUCTION)]

	#Normalize the features
	featuresVec = p_eigVec.dot(digit.T).flatten()
	maxV = featuresVec.max()
	minV = featuresVec.min()
	featuresVec = (2*(featuresVec - minV)/(maxV - minV)) - 1.0
	if np.isnan(featuresVec.mean()):
		featuresVec = np.nan_to_num(featuresVec)

	#Finally, use the numberClassifier to predict the value
	number = numberClassifier.predict([featuresVec])[0]
	return str(number)


def readHPMP_fromImage(image, numberClassifier):
	imgHeight, imgWidth = image.shape
	#Check Image shape
	if imgWidth<=0 or imgHeight<=0:
		return (-1, -1)
	#_______70%_____|___30%___|
	#---------------|---------|>>(h)
	#hpBarhpBarhpBar|HP DIGITS|
	#---------------|---------|>>(h/2)
	#mpBarmpBarmpBar|MP DIGITS|
	#---------------|---------|>>(0)
	image = image[0:imgHeight, int(0.70*imgWidth):imgWidth]
	
	#Resize in order to improve 'findBoundaryBoxes' and 'enhanceDigit' methods
	image = cv2.resize(image, (0,0), fx=10.0, fy=10.0)
	
	#Get the shape, in sequence split hpDigits and mpDigits
	h, w = image.shape
	hp_img = image[0:int(h/2), 0:w]
	mp_img = image[int(h/2):h, 0:w]

	#Find numbers contours
	bboxes = findBoundaryBoxes(hp_img)
	hp = []
	for bbox in bboxes:
		xmin, ymin, xmax, ymax = bbox
		digit = hp_img[ymin:ymax, xmin:xmax]
		digitValue = recognizeDigit(digit, numberClassifier)
		hp.append((xmin, digitValue))	
	hp.sort(key=lambda x: x[0])
	hp = [n[1] for n in hp]
	try:
		hp = int(''.join(hp))
	except Exception as e:
		hp = -1

	bboxes = findBoundaryBoxes(mp_img)
	mp = []
	for bbox in bboxes:
		xmin, ymin, xmax, ymax = bbox
		digit = mp_img[ymin:ymax, xmin:xmax]
		digitValue = recognizeDigit(digit, numberClassifier)
		mp.append((xmin, digitValue))
	mp.sort(key=lambda x: x[0])
	mp = [n[1] for n in mp]
	try:
		mp = int(''.join(mp))
	except Exception as e:
		mp = -1

	if hp>99999:
		hp = 99999
	
	if mp>99999:
		mp = 99999
	
	return (hp, mp)


if __name__ == '__main__':
	print('Script executed as __main__')
	sys.exit(0)
