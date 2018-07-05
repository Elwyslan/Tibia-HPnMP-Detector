import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import asyncio
import re
import sys

VERDANA16 = ("Verdana", 16)
VERDANA14 = ("Verdana", 14)
VERDANA12 = ("Verdana", 12)
VERDANA10 = ("Verdana", 10)
VERDANA09 = ("Verdana", 9)

class MaskEntry(threading.Thread):
	def __init__(self, entryWidget):
		threading.Thread.__init__(self)
		self.entryInstance = entryWidget
		self.runLoop = True
		self.daemon = True
		self.start()

	def run(self):
		while self.runLoop:
			txtValue = self.entryInstance.get()
			if len(txtValue) > 0:
				maskedValue = re.sub("[^0-9]", "", txtValue)

				if len(maskedValue) > 5:
					maskedValue = maskedValue[0:5]

				if maskedValue != txtValue:
					self.entryInstance.delete(0, tk.END)
					self.entryInstance.insert(0, maskedValue)

			time.sleep(0.1)

	def endLoop(self):
		self.runLoop = False

	def startLoop(self):
		self.runLoop = True


class HPMPdetectorScreen(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.geometry("450x75")
		self.resizable(0,0)
		self.title("Tibia - HP and MP detector")
		self.iconbitmap(self,default='app_favico.ico')

		mainFrame = tk.Frame(self)
		
		#Frame TOP
		frameTop = tk.Frame(mainFrame)
		hpInitLbl = tk.Label(frameTop, text="HP:", font=VERDANA14)
		self.hpScannedLbl = tk.Label(frameTop, font=VERDANA14, width=5, anchor=tk.E)
		hpThrsLbl = tk.Label(frameTop, text="HP thrs.:", font=VERDANA09, anchor=tk.W)
		self.hpThreshVar = tk.StringVar()
		hpThrsEnt = tk.Entry(frameTop, font=VERDANA09, width=6, textvariable=self.hpThreshVar)
		self.hpMskThread = MaskEntry(hpThrsEnt)
		hpActionLbl = tk.Label(frameTop, text="Action:", font=VERDANA09, anchor=tk.W)
		self.hpActionEnt = tk.Entry(frameTop, font=VERDANA09, width=10)
		self.hpHealCheckbox = tk.IntVar()
		hpCheckB = tk.Checkbutton(frameTop, font=VERDANA09, text="Enable HP\nAuto Heal", variable=self.hpHealCheckbox)
		#Pack Frame TOP
		hpInitLbl.grid(row=0,column=0)
		self.hpScannedLbl.grid(row=0,column=1)
		hpThrsLbl.grid(row=0,column=2)
		hpThrsEnt.grid(row=0,column=3)
		hpActionLbl.grid(row=0,column=4)
		self.hpActionEnt.grid(row=0,column=5)
		hpCheckB.grid(row=0,column=6)
		frameTop.pack(side=tk.TOP)

		#Frame BOTTOM
		frameBottom = tk.Frame(mainFrame)
		mpInitLbl = tk.Label(frameBottom, text="MP:", font=VERDANA14)
		self.mpScannedLbl = tk.Label(frameBottom, font=VERDANA14, width=5, anchor=tk.E)
		mpThrsLbl = tk.Label(frameBottom, text="MP thrs.:", font=VERDANA09, anchor=tk.S)
		self.mpThreshVar = tk.StringVar()
		mpThrsEnt = tk.Entry(frameBottom, font=VERDANA09, width=6, textvariable=self.mpThreshVar)
		self.mpMskThread = MaskEntry(mpThrsEnt)
		mpActionLbl = tk.Label(frameBottom, text="Action:", font=VERDANA09, anchor=tk.W)
		self.mpActionEnt = tk.Entry(frameBottom, font=VERDANA09, width=10)
		self.mpHealCheckbox = tk.IntVar()
		mpCheckB = tk.Checkbutton(frameBottom, font=VERDANA09, text="Enable MP\nAuto Heal", variable=self.mpHealCheckbox)
		#Pack Frame BOTTOM
		mpInitLbl.grid(row=0,column=0)
		self.mpScannedLbl.grid(row=0,column=1)
		mpThrsLbl.grid(row=0,column=2)
		mpThrsEnt.grid(row=0,column=3)
		mpActionLbl.grid(row=0,column=4)
		self.mpActionEnt.grid(row=0,column=5)
		mpCheckB.grid(row=0,column=6)
		frameBottom.pack(side=tk.BOTTOM)

		mainFrame.pack()

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


class MainApp(threading.Thread):
	def __init__(self, tkApp):
		threading.Thread.__init__(self)
		self.tkAppInstance = tkApp
		self.runLoop = True
		self.daemon = True
		self.start()

	def run(self):
		while self.runLoop:
			self.tkAppInstance.setHPLbl(random.randint(0,20000))
			self.tkAppInstance.setMPLbl(random.randint(0,20000))

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
		
		elapsedTime = 0.0
		while len(threading.enumerate())>1:
			time.sleep(0.01)
			elapsedTime += 0.01
			if elapsedTime>2.0:
				print('Running Theads: {}'.format(threading.enumerate()))
				break
		try:
			self.tkAppInstance.destroy()
		except Exception as e:
			print(str(e))
			sys.exit(0)

	def startLoop(self):
		self.runLoop = True



if __name__ == '__main__':
	print("Script executing as 'Main'")
	
	tkApp = HPMPdetectorScreen()
	
	auxThread = MainApp(tkApp)

	tkApp.protocol("WM_DELETE_WINDOW", auxThread.endLoop)
	
	tkApp.mainloop()
	
	if auxThread.isAlive():
		auxThread.endLoop()
		while auxThread.isAlive():
			continue

	print('End!')
	sys.exit(0)