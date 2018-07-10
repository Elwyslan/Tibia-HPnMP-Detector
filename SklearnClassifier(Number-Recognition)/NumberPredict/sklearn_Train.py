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
	for file in os.listdir(SRCPATH):
		sourceFilepath = SRCPATH+file
		index = int(file.split('_')[0])
		rawSamples[index].append(sourceFilepath)

	qtdSamples = []
	for i in range(len(rawSamples)):
		qtdSamples.append(len(rawSamples[i]))

	qtdSamples = min(qtdSamples)
	print('Qtd. min. samples: {}'.format(qtdSamples))

	for i in range(len(rawSamples)):
		while (len(rawSamples[i]) - qtdSamples)>0:
			random.shuffle(rawSamples[i])
			rawSamples[i].pop()

		print('({}) Qtd. samples: {}'.format(i, len(rawSamples[i])))

	X = []
	y = []
	for targetNumber, samples in enumerate(rawSamples):
		for sample in samples:
			print('Target:{} >> {}'.format(targetNumber, sample))
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
			p_eigVec = Vh[0:int(len(Vh)/EIGENVEC_REDUCTION)]
			
			#Normalize the features
			featuresVec = p_eigVec.dot(enhancedImg.T).flatten()
			maxV = featuresVec.max()
			minV = featuresVec.min()
			featuresVec = (2*(featuresVec - minV)/(maxV - minV)) - 1.0
			if np.isnan(featuresVec.mean()):
				featuresVec = np.nan_to_num(featuresVec)
			X.append(featuresVec)
			y.append(targetNumber)
	
	#Create dataset
	X = np.array(X, dtype=np.float32)
	y = np.array(y, dtype=np.int32)
	dataset = pd.DataFrame(X)
	dataset['target'] = pd.Series(y)

	#Cria o 'TARGET SET' para ser usado no Machine Learning
	y = dataset['target'].values

	print('\n\n\nDataset >> {}'.format(Counter(y)))
	dataset.drop(['target'], axis=1, inplace=True)
	X = dataset.values

	if len(X)!=len(y):
		raise Exception('\nDataSets with different sizes!\nTrainSet Size: {}\nTargetSet Size: {}'.format(len(X), len(y)))
	
	clf = VotingClassifier([('lsvc', svm.LinearSVC()),
							('knn', neighbors.KNeighborsClassifier()),
							('rfor', RandomForestClassifier())])
	
	
	print('Fitting Number Classifier....')
	clf.fit(X, y)
	print('End Fit!\nSaving classifier...')
	
	unixTimestamp = str(time.time()).replace('.','')
	with open(unixTimestamp+'_numberPredict.pkl', 'wb') as f:
		pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
		print('Classifier {} saved!'.format(unixTimestamp+'_numberPredict.pkl'))

	print('End!')
