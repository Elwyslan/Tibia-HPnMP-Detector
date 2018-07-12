import sys
import threading
import numpy as np
import time
from PIL import ImageGrab
from utils import readHPMP_fromImage
import cv2
import warnings
warnings.filterwarnings("ignore")

class ScreenshotGrab(threading.Thread):
	def __init__(self, cannyMin=100, cannyMax=200):
		threading.Thread.__init__(self)
		self.name = 'Screenshot Grab Thread'
		self.daemon = True
		#Flag to main loop on 'run' method
		self.runLoop = True
		self.cannyMin = cannyMin
		self.cannyMax = cannyMax
		self.screenCanny = None
		self.screenCap = None
		self.screenshotID = 0

	def run(self):
		#Set canny threshold
		cannyMinV = self.cannyMin
		cannyMaxV = self.cannyMax
		while self.runLoop:
			try:
				#Take one screenshot
				screenCap = ImageGrab.grab()
			except Exception as e:
				print(str(e))
			#Adjust and Store the screenshot
			screenCap = np.array(screenCap)
			self.screenCap = cv2.cvtColor(screenCap, cv2.COLOR_RGB2GRAY)
			self.screenCanny = cv2.Canny(self.screenCap, cannyMinV, cannyMaxV)
			self.screenshotID += 1

			#Release CPU for 15ms
			time.sleep(0.015)

	#Return the last Screenshot
	def getScreenShot(self, cannyImg=False):
		if cannyImg:
			return self.screenCanny
		else:
			return self.screenCap

	#Return TRUE if at least one screenshot has occur
	def hasScreenshotAvaliable(self):
		if self.screenshotID > 0:
			return True
		else:
			return False

	#End the main loop on 'run' fuction of this thread
	def endLoop(self):
		self.runLoop = False


class HPnMPBarsDetection(threading.Thread):
	def __init__(self, hpmpCascade, numberClassifier):
		threading.Thread.__init__(self)
		self.name = 'Automatic HPMP Detection Thread'
		self.daemon = True
		#Flag to main loop on 'run' method
		self.runLoop = True
		#Set the ScreenshotGrabber
		self.scrGrab = ScreenshotGrab()
		#Set the classifiers
		self.numberClassifier = numberClassifier
		self.hpmpCascade = hpmpCascade
		#Control variables
		self.processedImg = -1
		self.detectedBox = None
		self.detectedHP = -1
		self.detectedMP = -1

	def run(self):
		#Start the ScreenshotGrab thread
		self.scrGrab.start()
		
		#Wait 1st Screenshot occur
		while not self.scrGrab.hasScreenshotAvaliable():
			print('No screenshot avaliable until now')
			time.sleep(0.015)

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
			#Check if it is a new screenshot comparing his screenshot ID
			if self.processedImg != self.scrGrab.screenshotID:
				#Get the screenshot from scrGrab thread
				mainSS =  self.scrGrab.getScreenShot(cannyImg=False)
				cannySS = self.scrGrab.getScreenShot(cannyImg=True)
				#cannySS = cv2.resize(cannySS, (0,0), fx=scaleFWidth, fy=scaleFHeight)
				try:
					#Execute the Haar Cascade Classification
					hp_mp_bars = self.hpmpCascade.detectMultiScale(cannySS)
					#Process only the 1st detected object
					for (x, y, w, h) in hp_mp_bars:
						self.detectedBox = (x, y, w, h)
						#Extract the dected image from mainSS
						img = mainSS[y:y+h, x:x+w]
						hp, mp = readHPMP_fromImage(img, self.numberClassifier)
						self.detectedHP, self.detectedMP = hp, mp
						print('HP:{}\nMP:{}\n{}'.format(hp,mp,'#'*50))
						break
						
				except Exception as e:
					print(str(e))
				self.processedImg = self.scrGrab.screenshotID
			else:
				time.sleep(0.01)
				
		self.scrGrab.endLoop()

	#Return the last detected box with contains 'hp and mp bars'
	def getDetectedBox(self):
		if isinstance(self.detectedBox, tuple):
			return self.detectedBox
		else:
			return self.detectedBox

	#Return the last values of HP and MP tha has been detected
	def getHPMP(self, bbox=None):
		if isinstance(bbox, tuple):
			x, y, w, h = bbox
			img =  self.scrGrab.getScreenShot(cannyImg=False)
			img = img[y:y+h, x:x+w]
			hp, mp = readHPMP_fromImage(img, self.numberClassifier)
			return (hp, mp)
		else:
			return (self.detectedHP, self.detectedMP)
	
	#End the main loop on 'run' fuction of this and screenGrabber thread
	def endLoop(self):
		self.runLoop = False
		self.scrGrab.endLoop()
		


if __name__ == '__main__':
	print('Script executed as __main__')
	sys.exit(0)
