from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
import imutils
import sys
import os


        
class Root(Tk):

	def __init__(self):
		super(Root, self).__init__()
		self.title("Project Under Pattern Recognition")
		self.minsize(300, 400)
		self.configure(background='#F0F0F0')
		style = ttk.Style()
		style.configure('TLabelframe.Label', font=('Helvetica', 12, 'bold'))
		style.configure('TLabelframe.Label', foreground ='white')

		style.configure('lab1.TLabelframe.Label', background='#FF0067')
		style.configure('lab1.TLabelframe', background='#FF0067')

		style.configure('lab2.TLabelframe.Label', background='#00D1D6')
		style.configure('lab2.TLabelframe', background='#00D1D6')

		style.configure('lab3.TLabelframe.Label', background='#FFB900')
		style.configure('lab3.TLabelframe', background='#FFB900')

		style.configure('lab4.TLabelframe.Label', background='#9035C3')
		style.configure('lab4.TLabelframe', background='#9035C3')


		style.configure('Test.TLabel', background='#F0F0F0')

		style.configure('TButton', font=('Helvetica', 15))

		self.icon = PhotoImage(file='images/video.png')
		self.photoimage = self.icon.subsample(1, 1)

		self.camera = PhotoImage(file='images/camera.png')
		self.photocamera = self.camera.subsample(1, 1)

		self.gears = PhotoImage(file='images/gears.png')
		self.photogears = self.gears.subsample(1, 1)   

		self.ai = PhotoImage(file='images/ai.png')
		self.photoai = self.ai.subsample(1, 1)  

		

		self.labelFrame = ttk.LabelFrame(self, text ='Input Video', style="lab1.TLabelframe" )
		self.labelFrame.grid(column = 0, row = 0, padx = 5, pady = 5)
		
		self.labelFrame1 = ttk.LabelFrame(self, text ='Path', style="lab2.TLabelframe")
		self.labelFrame1.grid(column = 0, row = 1, padx = 5, pady = 10)

		self.labelFrame2 = ttk.LabelFrame(self, text ='Run',style="lab4.TLabelframe")
		self.labelFrame2.grid(column = 0, row = 2, padx = 5, pady = 5)

		self.labelFrame3 = ttk.LabelFrame(self, text ='Live Stream',  style="lab3.TLabelframe")
		self.labelFrame3.grid(column = 0, row = 3, padx = 5, pady = 5)

		self.label1 = ttk.Label(wraplength="4i", justify="right", anchor="n",padding=(10, 2, 10, 6),font=('Helvetica', 35), text='Human Activities Recognition',style= 'Test.TLabel')
		self.label1.grid(column = 0, row = 4, padx = 5, pady = 5)
		self.label1.config(image=self.photoai,compound = LEFT)

		self.button()
		
	def button(self):
		

		self.button = ttk.Button(self.labelFrame, text = '     Browse File', width=50, command = self.fileDialog)
		self.button.grid(column = 0, row = 0, ipady=10)
		self.button.config(image=self.photoimage,compound = LEFT)

		self.button1 = ttk.Button(self.labelFrame2, text = '  Run Program', width=50,command = self.RunPro)
		self.button1.grid(column = 0, row = 2,  ipady=10)
		self.button1.config(image=self.photogears,compound = LEFT)
		
		self.button2 = ttk.Button(self.labelFrame3, text = '      Live Recognition', width=50,command = self.livecam)
		self.button2.grid(column = 0, row = 3,  ipady=10)
		self.button2.config(image=self.photocamera,compound = LEFT)

		


	def livecam(self):
		amodel = 'resnet-34_kinetics.onnx'
		aclasses = 'action_recognition_kinetics.txt'
		vidpath = 0
		#vidpath = 'example_activities.mp4'
		# load the contents of the class labels file, then define the sample
		# duration (i.e., # of frames for classification) and sample size
		# (i.e., the spatial dimensions of the frame)
		CLASSES = open(aclasses).read().strip().split("\n")
		SAMPLE_DURATION = 16
		SAMPLE_SIZE = 112

		# load the human activity recognition model
		print("[INFO] loading human activity recognition model...")
		net = cv2.dnn.readNet(amodel)

		# grab a pointer to the input video stream
		print("[INFO] accessing video stream...")
		vs = cv2.VideoCapture(vidpath)

		# loop until we explicitly break from it
		while True:
			# initialize the batch of frames that will be passed through the
			# model
			frames = []

			# loop over the number of required sample frames
			for i in range(0, SAMPLE_DURATION):
				# read a frame from the video stream
				(grabbed, frame) = vs.read()

				# if the frame was not grabbed then we've reached the end of
				# the video stream so exit the script
				if not grabbed:
					print("[INFO] no frame read from stream - exiting")
					sys.exit(0)

				# otherwise, the frame was read so resize it and add it to
				# our frames list
				frame = imutils.resize(frame, width=600)
				frames.append(frame)

			# now that our frames array is filled we can construct our blob
			blob = cv2.dnn.blobFromImages(frames, 1.0,
				(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
				swapRB=True, crop=True)
			blob = np.transpose(blob, (1, 0, 2, 3))
			blob = np.expand_dims(blob, axis=0)

			# pass the blob through the network to obtain our human activity
			# recognition predictions
			net.setInput(blob)
			outputs = net.forward()
			label = CLASSES[np.argmax(outputs)]

			# loop over our frames
			for frame in frames:
				# draw the predicted activity on the frame
				cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
				cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
					0.8, (255, 255, 255), 2)

				# display the frame to our screen
				cv2.imshow("Activity Recognition", frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break

	def RunPro(self):
		
		amodel = 'models_n_classes/resnet-34_kinetics.onnx'
		aclasses = 'models_n_classes/action_recognition_kinetics.txt'
		vidpath = self.path
		#vidpath = 'example_activities.mp4'
		# load the contents of the class labels file, then define the sample
		# duration (i.e., # of frames for classification) and sample size
		# (i.e., the spatial dimensions of the frame)
		CLASSES = open(aclasses).read().strip().split("\n")
		SAMPLE_DURATION = 16
		SAMPLE_SIZE = 112

		# load the human activity recognition model
		print("[INFO] loading human activity recognition model...")
		net = cv2.dnn.readNet(amodel)

		# grab a pointer to the input video stream
		print("[INFO] accessing video stream...")
		vs = cv2.VideoCapture(vidpath)

		# loop until we explicitly break from it
		while True:
			# initialize the batch of frames that will be passed through the
			# model
			frames = []

			# loop over the number of required sample frames
			for i in range(0, SAMPLE_DURATION):
				# read a frame from the video stream
				(grabbed, frame) = vs.read()

				# if the frame was not grabbed then we've reached the end of
				# the video stream so exit the script
				if not grabbed:
					print("[INFO] no frame read from stream - exiting")
					sys.exit(0)

				# otherwise, the frame was read so resize it and add it to
				# our frames list
				frame = imutils.resize(frame, width=600)
				frames.append(frame)

			# now that our frames array is filled we can construct our blob
			blob = cv2.dnn.blobFromImages(frames, 1.0,
				(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
				swapRB=True, crop=True)
			blob = np.transpose(blob, (1, 0, 2, 3))
			blob = np.expand_dims(blob, axis=0)

			# pass the blob through the network to obtain our human activity
			# recognition predictions
			net.setInput(blob)
			outputs = net.forward()
			label = CLASSES[np.argmax(outputs)]

			# loop over our frames
			for frame in frames:
				# draw the predicted activity on the frame
				cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
				cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
					0.8, (255, 255, 255), 2)

				# display the frame to our screen
				cv2.imshow("Activity Recognition", frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break
	

	def fileDialog(self):

		self.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("Video files","*.mp4"),("all files","*.*")))
		print (self.filename)

		self.e1 = ttk.Entry(self.labelFrame1, width = 50)
		self.e1.insert(0, self.filename)
		self.e1.grid(row=1, column=0, columnspan=50, ipady=10)
		
		Root.OpenVideo(self.filename)
		#place Video
		
		self.path = self.filename
		print (self.path)
		
		'''im = Image.open(self.path)
		resized = im.resize((300, 300),Image.ANTIALIAS)
		tkimage = ImageTk.PhotoImage(resized)
		myvar=ttk.Label(self.labelFrame2,Video = tkVideo)
		myvar.image = tkimage
		myvar.grid(column=0, row=4)'''

	def OpenVideo(self):
		pass

if __name__ == '__main__':
    
    root = Root()
    
    root.mainloop()
