import sys
import threading
from PIL import ImageGrab
from PIL import Image
import cv2
import numpy as np
import time

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
	def __init__(self, hpmpCascade):
		threading.Thread.__init__(self)
		self.runLoop = True
		self.daemon = True
		self.name = 'Automatic HPMP Detection Thread'
		self.scrGrab = ScreenshotGrab()

		if isinstance(hpmpCascade, cv2.CascadeClassifier):
			self.hpmpCascade = hpmpCascade
		else:
			raise Exception('Invalid Cascade Classifier')

		self.processedImg = -1
		self.detectedBox = None

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
						print('detectedBox: {}',self.detectedBox)
						break
						
				except Exception as e:
					print(str(e))
				self.processedImg = self.scrGrab.screenshotID
				
		self.scrGrab.endLoop()

	def getDetectedBox(self):
		if isinstance(self.detectedBox, tuple):
			return self.detectedBox
		else:
			return (0, 0, 0, 0)
	
	def endLoop(self):
		self.runLoop = False




if __name__ == '__main__':
	print('Script executed as __main__')
	print('End!')
