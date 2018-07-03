import os
import cv2
import random
import shutil
import re
import numpy as np

POS_IMAGES_PATH = '$$$'
NEG_IMAGES_PATH = '$$$'

NUM_POS_SAMPLES = 2500
NUM_NEG_SAMPLES = 1200

if __name__ == '__main__':
	#Check sources path
	if not os.path.exists(POS_IMAGES_PATH) and not os.path.exists(NEG_IMAGES_PATH):
		print "Sourcepaths don't exist\nEnd script NOW!"
		quit()

	#Create folders 'pos' and 'neg'
	if not os.path.exists('pos'):
		os.mkdir('pos')
	else:
		shutil.rmtree('pos')
		os.mkdir('pos')

	if not os.path.exists('neg'):
		os.mkdir('neg')
	else:
		shutil.rmtree('neg')
		os.mkdir('neg')

	#Gen POS images from POS_IMAGES_PATH
	posSamples = []
	for file in os.listdir(POS_IMAGES_PATH):
		sourceFilepath = POS_IMAGES_PATH+file
		savePath = 'pos/'+file
		posSamples.append((sourceFilepath, savePath))

	#Store width and height for each image
	sumWidth = []
	sumHeight = []

	#Generate 'info.txt'
	with open('info.txt','w') as f:
		for index in range(NUM_POS_SAMPLES):
			random.shuffle(posSamples)
			sourceFilepath, savePath = posSamples.pop()
			try:
				img = cv2.imread(sourceFilepath, cv2.IMREAD_GRAYSCALE)
				cv2.imwrite(savePath, img)
				
				height, width = img.shape
				sumHeight.append(int(height))
				sumWidth.append(int(width))

				infoLine = savePath + ' 1 0 0 {} {}'.format(width, height)
				
				print infoLine
				f.write(infoLine+'\n')
			except Exception as e:
				print str(e)

	#Generate 'bg.txt'
	with open('bg.txt','w') as f:
		images = list(os.listdir(NEG_IMAGES_PATH))
		for index in range(NUM_NEG_SAMPLES):
			random.shuffle(images)
			img = images.pop()
			
			sourceFilepath = NEG_IMAGES_PATH+img
			img = cv2.imread(sourceFilepath, cv2.IMREAD_GRAYSCALE)
			
			try:
				print 'Saving neg/{}.png {}'.format(index,img.shape)
			except Exception as e:
				print str(e)
			
			cv2.imwrite('neg/{}.png'.format(index),img)
			
			f.write('neg/{}.png\n'.format(index))

	print 'Statistics\n'
	wSum = np.array(sumWidth)
	hSum = np.array(sumHeight)
	print 'width min={}, max={}, mean={}'.format(wSum.mean()-wSum.std(), wSum.mean()+wSum.std(), wSum.mean())
	print 'height min={}, max={}, mean={}'.format(hSum.mean()-hSum.std(), hSum.mean()+hSum.std(), hSum.mean())
	print 'width choose (w/10) mean={}'.format(wSum.mean()/10.0)
	print 'height choose (h/10) mean={}'.format(hSum.mean()/10.0)
	print 'Fim!'
	