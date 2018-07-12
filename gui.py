import threading
import tkinter as tk
import sys
import time
from screenshotWorker import HPnMPBarsDetection
from utils import getRawScreenshot
from PIL import Image, ImageTk
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

#Fonts 
VERDANA14 = ("Verdana", 14)
VERDANA09 = ("Verdana", 9)
VERDANA08 = ("Verdana", 8)


class DisplayImg(threading.Thread):
	def __init__(self, lblWidget, cascadeClassifier, numberClassifier, mainTkApp):
		threading.Thread.__init__(self)
		self.name = 'Display Image Thread'
		self.hpmpBarsDetector = HPnMPBarsDetection(cascadeClassifier, numberClassifier)
		self.hpmpBarsDetector.start()
		self.lblInstance = lblWidget
		self.mainTkApp = mainTkApp
		self.runLoop = True
		self.daemon = True
		self.maxWidth = 420
		self.maxHeight = 85
		self.lastBBox = (0, 0, 420, 85)

	def run(self):
		while self.runLoop:
			haarCheckB = self.mainTkApp.getHaarCheckBox()
			
			if haarCheckB == 1:
				#Haar Cascade Classifier bbox
				bbox = self.hpmpBarsDetector.getDetectedBox()
				fixSizeBbox = (self.maxWidth, self.maxHeight)
			else:
				#User input bbox
				bbox = self.mainTkApp.getBBox()
				fixSizeBbox = None
			
			try:
				x0 = int(bbox[0])
				y0 = int(bbox[1])
				w = int(bbox[2])
				h = int(bbox[3])
				if x0 < 0:
					x0 = 0
				if y0 < 0:
					y0 = 0
				if w < 0:
					w = 0
				if h < 0:
					h = 0
				if w > self.maxWidth or w <= 0:
					w = self.maxWidth
				if h > self.maxHeight or h <= 0:
					h = self.maxHeight

				bbox = (x0, y0, w, h)
			except Exception as e:
				bbox = (0, 0, self.maxWidth, self.maxHeight)
			
			img = getRawScreenshot(PILFormat=True, bbox=bbox, cv2ResizeBbox=fixSizeBbox)
			self.lastBBox = bbox
			img = ImageTk.PhotoImage(img)
			self.lblInstance.configure(image=img)
			self.lblInstance.image = img
			time.sleep(0.005)

	def getHPMP(self):
		haarCheckB = self.mainTkApp.getHaarCheckBox()
		if haarCheckB == 1:
			return self.hpmpBarsDetector.getHPMP()
		else:
			return self.hpmpBarsDetector.getHPMP(bbox=self.lastBBox)
	

	def endLoop(self):
		self.runLoop = False
		self.hpmpBarsDetector.endLoop()


class MaskEntry(threading.Thread):
	def __init__(self, entryWidget, maxLength=5):
		threading.Thread.__init__(self)
		self.name = 'Mask Entry Thread'
		self.entryInstance = entryWidget
		self.runLoop = True
		self.daemon = True
		self.maxLength =  maxLength

	def run(self):
		while self.runLoop:
			try:
				txtValue = self.entryInstance.get()
				if len(txtValue) > 0:
					maskedValue = re.sub("[^0-9]", "", txtValue)

					if len(maskedValue) > self.maxLength:
						maskedValue = maskedValue[0:self.maxLength]

					if maskedValue != txtValue:
						self.entryInstance.delete(0, tk.END)
						self.entryInstance.insert(0, maskedValue)
			
			except Exception as e:
				print('Mask Entry exception: {}'.format(str(e)))
			finally:
				time.sleep(0.1)


	def endLoop(self):
		self.runLoop = False


class GUIManager(threading.Thread):
	def __init__(self, tkApp):
		threading.Thread.__init__(self)
		self.name = 'Gui Manager Thread'
		self.tkAppInstance = tkApp
		self.runLoop = True
		self.daemon = True
		self.start()

	def run(self):
		#MAIN LOOP
		self.tkAppInstance.displayImg.start()
		self.tkAppInstance.hpMskThread.start()
		self.tkAppInstance.mpMskThread.start()
		self.tkAppInstance.x0MskThread.start()
		self.tkAppInstance.y0MskThread.start()
		self.tkAppInstance.hThrsMskThread.start()
		self.tkAppInstance.wThrsMskThread.start()

		time.sleep(1)
		while self.runLoop:
			hp, mp = self.tkAppInstance.displayImg.getHPMP()
			self.tkAppInstance.setHPLbl(hp)
			self.tkAppInstance.setMPLbl(mp)

			hpCheck = self.tkAppInstance.getHPCheckBox()
			mpCheck = self.tkAppInstance.getMPCheckBox()

			waitCooldown = False
			if hpCheck==1:
				hpV = self.tkAppInstance.getHPValue()
				hpTresh = self.tkAppInstance.getHPThresh()
				if hpV<=hpTresh:
					hpHealAct = self.tkAppInstance.getHPHealAct()
					print("# HEAL HP! # Pressing '{}'".format(hpHealAct))
					waitCooldown = True
					
			if mpCheck==1:
				mpV = self.tkAppInstance.getMPValue()
				mpTresh = self.tkAppInstance.getMPThresh()
				if mpV<=mpTresh:
					mpHealAct = self.tkAppInstance.getMPHealAct()
					print("# HEAL MP! # Pressing '{}'".format(mpHealAct))
					waitCooldown = True

			if waitCooldown:
				time.sleep(random.randint(2,3))
			else:
				time.sleep(0.1)

	def endLoop(self):
		self.runLoop = False
		self.tkAppInstance.hpMskThread.endLoop()
		self.tkAppInstance.mpMskThread.endLoop()
		self.tkAppInstance.x0MskThread.endLoop()
		self.tkAppInstance.y0MskThread.endLoop()
		self.tkAppInstance.hThrsMskThread.endLoop()
		self.tkAppInstance.wThrsMskThread.endLoop()
		self.tkAppInstance.displayImg.hpmpBarsDetector.endLoop()
		self.tkAppInstance.displayImg.endLoop()


#Main GUI Class 
class HPMPdetectorScreen(tk.Tk):

	def __init__(self, cascadeClassifier, numberClassifier, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.geometry("450x250")
		self.resizable(0,0)
		self.title("Tibia - HP and MP detector")
		self.iconbitmap(self,default='app_favico.ico')

		#START MAIN SCREEN CONSTRUCTION - mainFrame -
		mainFrame = tk.Frame(self)
		
		#Frame HP_Grid
		frameHP = tk.Frame(mainFrame)
		hpInitLbl = tk.Label(frameHP, text="HP:", font=VERDANA14)
		self.hpScannedLbl = tk.Label(frameHP, font=VERDANA14, width=5, anchor=tk.E)
		hpThrsLbl = tk.Label(frameHP, text="HP thrs.:", font=VERDANA09, anchor=tk.W)
		self.hpThreshVar = tk.StringVar()
		hpThrsEnt = tk.Entry(frameHP, font=VERDANA09, width=6, textvariable=self.hpThreshVar)
		self.hpMskThread = MaskEntry(hpThrsEnt, maxLength=5)
		hpActionLbl = tk.Label(frameHP, text="Action:", font=VERDANA09, anchor=tk.W)
		self.hpActionEnt = tk.Entry(frameHP, font=VERDANA09, width=10)
		self.hpHealCheckbox = tk.IntVar()
		hpCheckB = tk.Checkbutton(frameHP, font=VERDANA09, text="Enable HP\nAuto Heal", variable=self.hpHealCheckbox)
		#Pack FrameHP
		hpInitLbl.grid(row=0,column=0)
		self.hpScannedLbl.grid(row=0,column=1)
		hpThrsLbl.grid(row=0,column=2)
		hpThrsEnt.grid(row=0,column=3)
		hpActionLbl.grid(row=0,column=4)
		self.hpActionEnt.grid(row=0,column=5)
		hpCheckB.grid(row=0,column=6)

		#Frame MP_Grid
		frameMP = tk.Frame(mainFrame)
		mpInitLbl = tk.Label(frameMP, text="MP:", font=VERDANA14)
		self.mpScannedLbl = tk.Label(frameMP, font=VERDANA14, width=5, anchor=tk.E)
		mpThrsLbl = tk.Label(frameMP, text="MP thrs.:", font=VERDANA09, anchor=tk.S)
		self.mpThreshVar = tk.StringVar()
		mpThrsEnt = tk.Entry(frameMP, font=VERDANA09, width=6, textvariable=self.mpThreshVar)
		self.mpMskThread = MaskEntry(mpThrsEnt, maxLength=5)
		mpActionLbl = tk.Label(frameMP, text="Action:", font=VERDANA09, anchor=tk.W)
		self.mpActionEnt = tk.Entry(frameMP, font=VERDANA09, width=10)
		self.mpHealCheckbox = tk.IntVar()
		mpCheckB = tk.Checkbutton(frameMP, font=VERDANA09, text="Enable MP\nAuto Heal", variable=self.mpHealCheckbox)
		#Pack FrameMP
		mpInitLbl.grid(row=0,column=0)
		self.mpScannedLbl.grid(row=0,column=1)
		mpThrsLbl.grid(row=0,column=2)
		mpThrsEnt.grid(row=0,column=3)
		mpActionLbl.grid(row=0,column=4)
		self.mpActionEnt.grid(row=0,column=5)
		mpCheckB.grid(row=0,column=6)
		
		#Frame Image_Grid
		frameImg = tk.Frame(mainFrame)
		#Image parameters
		frameImgParams = tk.Frame(frameImg)
		self.HaarCascadecCheckbox = tk.IntVar()
		haarCheckB = tk.Checkbutton(frameImgParams, font=VERDANA08, text="Enable HaarCascade\nDetection", variable=self.HaarCascadecCheckbox)
		x0Lbl = tk.Label(frameImgParams, text="x0:", font=VERDANA08)
		self.x0Var = tk.StringVar()
		x0Ent = tk.Entry(frameImgParams, font=VERDANA08, width=4, textvariable=self.x0Var)
		self.x0MskThread = MaskEntry(x0Ent, maxLength=3)
		y0Lbl = tk.Label(frameImgParams, text="y0:", font=VERDANA08)
		self.y0Var = tk.StringVar()
		y0Ent = tk.Entry(frameImgParams, font=VERDANA08, width=4, textvariable=self.y0Var)
		self.y0MskThread = MaskEntry(y0Ent, maxLength=3)
		widthThrsLbl = tk.Label(frameImgParams, text="width:", font=VERDANA08)
		self.widthThrs = tk.StringVar()
		widthThrsEnt = tk.Entry(frameImgParams, font=VERDANA08, width=4, textvariable=self.widthThrs)
		self.wThrsMskThread = MaskEntry(widthThrsEnt, maxLength=3)
		heightThrsLbl = tk.Label(frameImgParams, text="height:", font=VERDANA08)
		self.heightThrs = tk.StringVar()
		heightThrsEnt = tk.Entry(frameImgParams, font=VERDANA08, width=4, textvariable=self.heightThrs)
		self.hThrsMskThread = MaskEntry(heightThrsEnt, maxLength=3)
		haarCheckB.grid(row=0,column=0)
		x0Lbl.grid(row=0,column=1)
		x0Ent.grid(row=0,column=2)
		y0Lbl.grid(row=0,column=3)
		y0Ent.grid(row=0,column=4)
		widthThrsLbl.grid(row=0,column=5)
		widthThrsEnt.grid(row=0,column=6)
		heightThrsLbl.grid(row=0,column=7)
		heightThrsEnt.grid(row=0,column=8)
		frameImgParams.grid(row=0,column=0)
		#Image show
		frameImgShow = tk.Frame(frameImg)
		imshowLbl = tk.Label(frameImgShow, borderwidth=2, relief="solid")
		self.displayImg = DisplayImg(imshowLbl, cascadeClassifier, numberClassifier, self)
		imshowLbl.pack()
		frameImgShow.grid(row=1,column=0)

		#mainframe Pack
		frameHP.grid(row=0,column=0)
		frameMP.grid(row=1,column=0)
		frameImg.grid(row=2,column=0)
		mainFrame.pack()
		
		self.GUIManager = GUIManager(self)
		
		self.protocol("WM_DELETE_WINDOW", self.terminateGUI)

	def setHPLbl(self, hpValue):
		self.hpScannedLbl['text'] = str(hpValue)

	def setMPLbl(self, mpValue):
		self.mpScannedLbl['text'] = str(mpValue)

	def getHPCheckBox(self):
		return self.hpHealCheckbox.get()

	def getMPCheckBox(self):
		return self.mpHealCheckbox.get()

	def getHPValue(self):
		try:
			return int(self.hpScannedLbl['text'])
		except Exception as e:
			print(str(e))
			return 0

	def getMPValue(self):
		try:
			return int(self.mpScannedLbl['text'])
		except Exception as e:
			print(str(e))
			return 0

	def getHPThresh(self):
		try:
			return int(self.hpThreshVar.get())
		except Exception as e:
			print(str(e))
			return 0

	def getMPThresh(self):
		try:
			return int(self.mpThreshVar.get())
		except Exception as e:
			print(str(e))
			return 0

	def getHPHealAct(self):
		return self.hpActionEnt.get()

	def getMPHealAct(self):
		return self.mpActionEnt.get()

	def getBBox(self):
		x0 = self.x0Var.get()
		y0 = self.y0Var.get()
		w = self.widthThrs.get()
		h = self.heightThrs.get()
		return (x0, y0, w, h)

	def getHaarCheckBox(self):
		return self.HaarCascadecCheckbox.get()

	def terminateGUI(self):
		self.GUIManager.endLoop()
		
		elapsedTime = 0.0
		while len(threading.enumerate())>1:
			time.sleep(0.01)
			elapsedTime += 0.01
			if elapsedTime>1.5:
				print('Running Theads: {}'.format(threading.enumerate()))
				break
		try:
			self.destroy()
		except Exception as e:
			print(str(e))
			sys.exit(0)



if __name__ == '__main__':
	print('Script executed as __main__')
	sys.exit(0)
