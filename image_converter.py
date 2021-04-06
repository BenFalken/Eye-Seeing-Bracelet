import os
import numpy as np
from PIL import Image
import pickle as pkl

class ImageConverter:
	def __init__(self):
		#self.PATH = '/home/pi/Desktop/Bracelet/'
		self.PATH = '/users/benfalken/desktop/SideProjectSadness/'

		self.dimen = 350
		self.imageNum = 200

		self.foldernames = ['SmoothSurfaces', 'RoughSurfaces']
		self.imageTypeNum = len(self.foldernames)

		self.allImages = np.zeros((self.imageNum, self.dimen, self.dimen))
		self.allLabels = np.zeros((self.imageNum, self.imageTypeNum))

		self.validImageNum = 0
		
		self.PICKLE_FILE_NAME = 'images_and_labels'
		self.PICKLE_FILE_PATH = self.PATH + self.PICKLE_FILE_NAME

	def process_all_images(self):
		retrieveData = input("Retrieve old data from the image archives?")
		if retrieveData == 't':
			return self.return_unpickled_data()
		for foldername in self.foldernames:
			path = self.PATH+foldername
			label = self.foldernames.index(foldername)

			for filename in os.listdir(path):
				try:
					
					data = self.convert(path, filename)

					self.allImages[self.validImageNum] = data
					self.allLabels[self.validImageNum][label] = 1

					self.validImageNum += 1
				except Exception as e:
					print('Conversion Failed: ' + filename)
					#print(e)
		self.purify_images()
		
		data = {
            "trainImages": self.trainImages,
            "trainLabels": self.trainLabels,
            "validImageNum": self.validImageNum
        }
		
		self.pickle_data(data)
		
		return self.trainImages, self.trainLabels, self.validImageNum

	def convert(self, path, filename):
		im = Image.open(os.path.join(path, filename));

		width, height = im.size

		if width < height:
			im = im.crop((0, int(round((height-width)/2)), width, int(round((height+width)/2))))
		elif height < width:
			im = im.crop((int(round((width-height)/2)), 0, int(round((height+width)/2)), height))
					
		im = im.resize((self.dimen, self.dimen))
		im = im.convert('L')
		data = np.asarray(im, dtype="int32")
		return data

	def purify_images(self):
		self.trainImages = np.zeros((self.validImageNum, self.dimen, self.dimen))
		self.trainLabels = np.zeros((self.validImageNum, self.imageTypeNum))

		for i in range(0, self.validImageNum):
			self.trainImages[i] = self.allImages[i]
			self.trainLabels[i] = self.allLabels[i]
			
	def pickle_data(self, data):
		file = open(self.PICKLE_FILE_PATH, 'wb')
		pkl.dump(data, file)
		file.close()
		
	def return_unpickled_data(self):
		file = open(self.PICKLE_FILE_PATH, "rb")
		data = pkl.load(file)
		return data["trainImages"], data["trainLabels"], data["validImageNum"]