import cv2
import os
import sys
import pickle
import time
import numpy as np
from PIL import ImageGrab
from utils import findBoundaryBoxes, enhanceDigit

EIGENVEC_REDUCTION = 1.4
CWD = os.getcwd()
SRC_HPMP_CASCADE = CWD + '\\0-HaarCascadeClassifier_(HPnMP-bars)\\results_w40h10\\cascade.xml'
SRC_NUMBER_CLASSIFIER = CWD + '\\1-SklearnClassifier(Number-Recognition)\\NumberPredict\\'
if sys.platform.startswith('linux'):
	SRC_HPMP_CASCADE = SRC_HPMP_CASCADE.replace('\\','/')
	SRC_NUMBER_CLASSIFIER = SRC_NUMBER_CLASSIFIER.replace('\\','/')
SRC_NUMBER_CLASSIFIER = SRC_NUMBER_CLASSIFIER + \
						[f for f in os.listdir(SRC_NUMBER_CLASSIFIER) if f.endswith('.pkl')][0]

#Load HAAR CASCADE
hp_mp_cascade = cv2.CascadeClassifier(SRC_HPMP_CASCADE)

#load NUMBER CLASSIFIER
numberClf = None
with open(SRC_NUMBER_CLASSIFIER, 'rb') as f:
	numberClf = pickle.load(f)

if __name__ == '__main__':
	print(SRC_HPMP_CASCADE)
	print(SRC_NUMBER_CLASSIFIER)


	print('End!')




	"""
	while True:
		screenCap = ImageGrab.grab()
		screenCap = np.array(screenCap)
		screenCap = cv2.cvtColor(screenCap, cv2.COLOR_RGB2BGR)
		screenHeight, screenWidth, _ = screenCap.shape
		
		#screenCap = cv2.resize(screenCap, (0,0), fx=0.75, fy=0.75)
		screenCap = cv2.resize(screenCap, (0,0), fx=0.55, fy=0.55)
		
		screenCanny = cv2.Canny(screenCap, 100, 200)

		hp_mp_bars = hp_mp_cascade.detectMultiScale(screenCanny)
		try:
			print('Qtd. deteccoes: {}'.format(len(hp_mp_bars)))
			unixTimestamp = str(time.time()).replace('.','')
			for (x, y, w, h) in hp_mp_bars:
				#print(x, y, w, h)
				savBox = screenCap[y:y+h, x:x+w]
				cv2.imwrite('savedBoxes/{}_{}.{}.{}.{}.png'.format(unixTimestamp,x,y,w,h), savBox)
				cv2.rectangle(screenCap, (x, y), (x+w, y+h), (0, 255, 0), 3)

		except Exception as e:
				print(str(e))
				pass
		
		cv2.imshow('Screenshot', screenCap)
		
		k = cv2.waitKey(5000) & 0xff
		if k==27:
			break"""
			