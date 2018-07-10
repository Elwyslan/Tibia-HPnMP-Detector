import cv2
import os
import sys
import pickle
import time
import numpy as np
from utils import findBoundaryBoxes, enhanceDigit
from gui import HPMPdetectorScreen

EIGENVEC_REDUCTION = 1.4
CWD = os.getcwd()
SRC_HPMP_CASCADE = CWD + '\\HaarCascadeClassifier_(HPnMP-bars)\\results_w40h10\\cascade.xml'
SRC_NUMBER_CLASSIFIER = CWD + '\\SklearnClassifier(Number-Recognition)\\NumberPredict\\'
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

	mainScreen = HPMPdetectorScreen(cascadeClassifier=hp_mp_cascade, numberClassifier=numberClf)
	mainScreen.mainloop()

	print('End!')
	sys.exit(0)
			