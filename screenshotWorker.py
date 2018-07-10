import sys
import os
import threading
from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np
import time
from skimage import measure
import random
import pandas as pd
from collections import Counter
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

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

def enhanceDigit(digitImage, thrOffset=50, scaleFactor=0.2):
	hist, _ = np.histogram(digitImage, 256, [0,256])
	indexMaxValue = np.argmax(hist)
	binaryThreshold = indexMaxValue + thrOffset
	_ ,thrDigit = cv2.threshold(digitImage, binaryThreshold, 255, cv2.THRESH_BINARY_INV)

	if scaleFactor > 0:
		h, w = thrDigit.shape
		thrDigit = cv2.resize(thrDigit, (0,0), fx=scaleFactor, fy=scaleFactor)

	return thrDigit

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


class ScreenshotGrab(threading.Thread):
	def __init__(self, cannyMin=100, cannyMax=200):
		threading.Thread.__init__(self)
		self.runLoop = True
		self.daemon = True
		self.name = 'Screenshot Thread'
		self.screenCap = None
		self.cannyMin = cannyMin
		self.cannyMax = cannyMax
		self.screenCanny = None
		self.screenshotID = 0

	def run(self):
		cannyMinV = self.cannyMin
		cannyMaxV = self.cannyMax
		while self.runLoop:
			try:
				screenCap = ImageGrab.grab()
			except Exception as e:
				print(str(e))
			screenCap = np.array(screenCap)
			self.screenCap = cv2.cvtColor(screenCap, cv2.COLOR_RGB2GRAY)
			self.screenCanny = cv2.Canny(self.screenCap, cannyMinV, cannyMaxV)
			self.screenshotID += 1

	def getScreenShot(self, cannyImg=False):
		if cannyImg:
			return self.screenCanny
		else:
			return self.screenCap

	def hasScreenshotAvaliable(self):
		if self.screenshotID > 0:
			return True
		else:
			return False

	def endLoop(self):
		self.runLoop = False


class HPnMPBarsDetection(threading.Thread):
	def __init__(self, hpmpCascade, numberClassifier):
		threading.Thread.__init__(self)
		self.runLoop = True
		self.daemon = True
		self.name = 'Automatic HPMP Detection Thread'
		self.scrGrab = ScreenshotGrab()

		self.numberClassifier = numberClassifier
		self.hpmpCascade = hpmpCascade

		self.processedImg = -1
		self.detectedBox = None
		self.detectedHP = -1
		self.detectedMP = -1

	def run(self):
		self.scrGrab.start()
		
		#Wait 1st Screenshot
		while not self.scrGrab.hasScreenshotAvaliable():
			print('No screenshot avaliable until now')
			time.sleep(0.1)

		#Get screen size and calculate the scale factor
		screenCap = self.scrGrab.getScreenShot()
		screenHeight, screenWidth = screenCap.shape
		estimatedBoxHeight = (35.0/1080.0)*screenHeight
		estimatedBoxWidth = (165.0/1920.0)*screenWidth
		origWidth, origHeight = self.hpmpCascade.getOriginalWindowSize()
		scaleFHeight = (screenHeight * origHeight)/estimatedBoxHeight
		scaleFWidth = (screenWidth * origWidth)/estimatedBoxWidth
		scaleFHeight = scaleFHeight/screenHeight
		scaleFWidth = scaleFWidth/screenWidth

		#start main loop
		while self.runLoop:
			if self.processedImg != self.scrGrab.screenshotID:
				mainSS =  self.scrGrab.getScreenShot(cannyImg=False)
				cannySS = self.scrGrab.getScreenShot(cannyImg=True)
				adjustedSS = cv2.resize(cannySS, (0,0), fx=scaleFWidth, fy=scaleFHeight)
				try:
					#hp_mp_bars = self.hpmpCascade.detectMultiScale(adjustedSS)
					hp_mp_bars = self.hpmpCascade.detectMultiScale(cannySS)
					for (x, y, w, h) in hp_mp_bars:
						self.detectedBox = (x, y, w, h)
						#Detect Numbers
						print('detectedBox: {}'.format(self.detectedBox))
						img = mainSS[y:y+h, x:x+w]
						hp_img, mp_img = self.extractHPMPImages(img)

						bboxesHP = findBoundaryBoxes(hp_img)
						hp = []
						for bbox in bboxesHP:
							xmin, ymin, xmax, ymax = bbox
							digit = hp_img[ymin:ymax, xmin:xmax]
							digitValue = self.recognizeDigit(digit)
							hp.append((xmin, digitValue))	
						hp.sort(key=lambda x: x[0])
						hp = [n[1] for n in hp]
						hp = int(''.join(hp))
						self.detectedHP = hp
						
						bboxesMP = findBoundaryBoxes(mp_img)
						mp = []
						for bbox in bboxesMP:
							xmin, ymin, xmax, ymax = bbox
							digit = mp_img[ymin:ymax, xmin:xmax]
							digitValue = self.recognizeDigit(digit)
							mp.append((xmin, digitValue))
						mp.sort(key=lambda x: x[0])
						mp = [n[1] for n in mp]
						mp = int(''.join(mp))
						self.detectedMP = mp

						print('HP:{}\nMP:{}\n{}'.format(hp,mp,'#'*50))
						break
						
				except Exception as e:
					print(str(e))
				self.processedImg = self.scrGrab.screenshotID
				
		self.scrGrab.endLoop()

	def extractHPMPImages(self, img):
		imgHeight, imgWidth = img.shape
		img = img[0:imgHeight, int(0.70*imgWidth):imgWidth]
		img = cv2.resize(img, (0,0), fx=10.0, fy=10.0)
		h, w = img.shape
		hp_img = img[0:int(h/2), 0:w]
		mp_img = img[int(h/2):h, 0:w]
		return (hp_img, mp_img)

	def recognizeDigit(self, digit):
		digit = cv2.resize(digit, (50, 65))
		digit = enhanceDigit(digit)
		#EXTRACT FEATURES
		#Mean normalize all rows
		digit = digit.astype(np.float32)
		for row in range(digit.shape[0]):
			u = digit[row, :].mean()
			digit[row, :] = digit[row, :] - u

		#Extract main eigenvectors
		_, _, Vh = np.linalg.svd(digit)
		p_eigVec = Vh[0:int(len(Vh)/1.4)]

		#Normalize the features
		featuresVec = p_eigVec.dot(digit.T).flatten()
		maxV = featuresVec.max()
		minV = featuresVec.min()
		featuresVec = (2*(featuresVec - minV)/(maxV - minV)) - 1.0
		if np.isnan(featuresVec.mean()):
			featuresVec = np.nan_to_num(featuresVec)

		number = self.numberClassifier.predict([featuresVec])[0]
		return str(number)


	def getDetectedBox(self):
		if isinstance(self.detectedBox, tuple):
			return self.detectedBox
		else:
			return (0, 0, 0, 0)

	def getHPMP(self):
		return (self.detectedHP, self.detectedMP)
	
	def endLoop(self):
		self.scrGrab.endLoop()
		self.runLoop = False


if __name__ == '__main__':
	print('Script executed as __main__')
	print('End!')
