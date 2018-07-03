import cv2
import os
import sys
from utils import findBoundaryBoxes

EIGENVEC_REDUCTION = 1.4
CWD = os.getcwd()
SRC_HPMP_CASCADE = CWD + '\\0-HaarCascadeClassifier_(HPnMP-bars)\\results_w40h10\\cascade.xml'
SRC_NUMBER_CLASSIFIER = CWD + '\\1-SklearnClassifier(Number-Recognition)\\NumberPredict\\'
if sys.platform.startswith('linux'):
	SRC_HPMP_CASCADE = SRC_HPMP_CASCADE.replace('\\','/')
	SRC_NUMBER_CLASSIFIER = SRC_NUMBER_CLASSIFIER.replace('\\','/')
SRC_NUMBER_CLASSIFIER = SRC_NUMBER_CLASSIFIER + \
						[f for f in os.listdir(SRC_NUMBER_CLASSIFIER) if f.endswith('.pkl')][0]


if __name__ == '__main__':
	print(SRC_HPMP_CASCADE)
	print(SRC_NUMBER_CLASSIFIER)
	print('End!')