import cv2
import os
import sys
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import pickle
import time
import warnings

EIGENVEC_REDUCTION = 1.4
CWD = os.getcwd()
SRCPATH = '{}\\numbers\\'.format(CWD)
if sys.platform.startswith('linux'):
	SRCPATH = SRCPATH.replace('\\','/')

def enhanceDigit(digitImage, thrOffset=50, scaleFactor=0.2):
	hist, _ = np.histogram(digitImage, 256, [0,256])
	indexMaxValue = np.argmax(hist)
	binaryThreshold = indexMaxValue + thrOffset
	_ ,thrDigit = cv2.threshold(digitImage, binaryThreshold, 255, cv2.THRESH_BINARY_INV)

	if scaleFactor > 0:
		h, w = thrDigit.shape
		thrDigit = cv2.resize(thrDigit, (0,0), fx=scaleFactor, fy=scaleFactor)

	return thrDigit

if __name__ == '__main__':
	#Ignore WARNINGS
	warnings.filterwarnings("ignore")
	srcPkl = CWD[:-17]+'NumberPredict\\'
	srcNumbers = srcPkl+'\\numbers\\'
	if sys.platform.startswith('linux'):
		srcPkl = srcPkl.replace('\\','/')
		srcNumbers = srcNumbers.replace('\\','/')

	pklClassifier = [f for f in os.listdir(srcPkl) if f.endswith('.pkl')][0]
	clfFile = srcPkl+pklClassifier

	classifier = None
	with open(clfFile, 'rb') as f:
		classifier = pickle.load(f)

	rawSamples = [[],
				  [],
				  [],
				  [],
				  [],
				  [],
				  [],
				  [],
				  [],
				  []]

	for file in os.listdir(srcNumbers):
		sourceFilepath = srcNumbers+file
		index = int(file.split('_')[0])
		rawSamples[index].append(sourceFilepath)
	for i in range(len(rawSamples)):
		random.shuffle(rawSamples[i])

	for targetNumber, samples in enumerate(rawSamples):
		results = []
		for sample in samples:
			srcImg = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
			enhancedImg = enhanceDigit(srcImg)

			#EXTRACT FEATURES
			#Mean normalize all rows
			enhancedImg = enhancedImg.astype(np.float32)
			for row in range(enhancedImg.shape[0]):
				u = enhancedImg[row, :].mean()
				enhancedImg[row, :] = enhancedImg[row, :] - u

			#Extract main eigenvectors
			_, _, Vh = np.linalg.svd(enhancedImg)
			p_eigVec = Vh[0:int(len(Vh)/1.4)]
			
			#Normalize the features
			featuresVec = p_eigVec.dot(enhancedImg.T).flatten()
			maxV = featuresVec.max()
			minV = featuresVec.min()
			featuresVec = (2*(featuresVec - minV)/(maxV - minV)) - 1.0
			if np.isnan(featuresVec.mean()):
				featuresVec = np.nan_to_num(featuresVec)
			
			results.append(classifier.predict([featuresVec])[0]==targetNumber)
			#print(targetNumber, sample)
		
		countDict = dict(Counter(results))
		correctPredictions = int(countDict[True])
		wrongPredictions = int(countDict[False])
		totalSamples = correctPredictions + wrongPredictions
		print('Target: {} ({} samples)\nCorrects Pred.: {}\nWrong Pred.: {}\nHit Rate: {}%\n{}'. format(targetNumber,
																										totalSamples,
																										correctPredictions,
																										wrongPredictions,
																										int((correctPredictions/totalSamples)*100.00),
																										'#'*30))

	print('End!')
