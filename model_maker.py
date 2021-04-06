import math, os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from PIL import Image
#from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, misc, ndimage
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import CubicSpline
from image_converter import ImageConverter
from camera_runner import CameraRunner

class Model:
	def __init__(self):
		self.camera_runner = CameraRunner(self)
		#self.PATH = '/home/pi/Desktop/Bracelet/'
		self.PATH = '/users/benfalken/desktop/SideProjectSadness/'

		#self.camera_runner = 0

		self.converter = ImageConverter()
		self.images, self.labels, valid_image_number = self.converter.process_all_images()
		self.images = self.images.astype("int32")

		self.STEP_VAL = 4
		self.NUM_IMAGES = int(self.labels.size/(2*self.STEP_VAL))

		self.smooth_r_vals = np.zeros((valid_image_number))
		self.rough_r_vals = np.zeros((valid_image_number))

		self.PICKLE_FILE_NAME = "final_model"
		self.PICKLE_FILE_PATH = self.PATH + self.PICKLE_FILE_NAME

	def start(self):
		choice = input("Create Model (c) or Run Model? (r) ")
		if choice == 'c':
			self.create_model()
		else:
			self.run_model()

	def run_model(self):
		self.unpickle_data()
		self.camera_runner.run()

	def create_model(self):
		num_images_counted, num_images_processed = 0, 0
		for im in self.images:
			if num_images_counted%self.STEP_VAL == 0:
				try:
					label = np.where(self.labels[num_images_counted] == 1)
					self.process_image(im, num_images_processed, label, is_individual_image=False)
					num_images_processed += 1
				except Exception as e:
					print(e)
					continue
			num_images_counted += 1
		self.create_data()
		self.pickle_data()

	def process_image(self, image, image_index, label_index, is_individual_image):
		print(image)
		max_grayscale_val, min_grayscale_val = np.max(image), np.min(image)
		n = max_grayscale_val - min_grayscale_val
		
		grayscale_vals = np.arange(min_grayscale_val, max_grayscale_val, 1)
		grayscale_val_freq = np.zeros((n))

		for i in range(n):
			grayscale_val_freq[i] = np.count_nonzero(image == min_grayscale_val+i)

		auto_corr = np.correlate(grayscale_val_freq, grayscale_val_freq, mode="full")
		x = np.arange(0, auto_corr.size)
		shapiro_test = stats.shapiro(auto_corr)
		r_mean = shapiro_test[1]

		if (not is_individual_image):
			im_type = int(label_index[0][0])
			self.record_image_properties(im_type, image_index, r_mean)
		else:
			return r_mean

	def record_image_properties(self, im_type, image_index, r_mean):
		if self.converter.foldernames[im_type] == 'SmoothSurfaces':
			self.smooth_r_vals[image_index] = r_mean
		else:
			self.rough_r_vals[image_index] = r_mean

	def create_data(self):

		self.smooth_r_mean = np.mean(self.smooth_r_vals)
		self.smooth_r_stdev = np.std(self.smooth_r_vals)

		self.rough_r_mean = np.mean(self.rough_r_vals)
		self.rough_r_stdev = np.std(self.rough_r_vals)

		self.data = {
			"smooth_r_mean": self.smooth_r_mean,
			"smooth_r_stdev": self.smooth_r_stdev,
			"rough_r_mean": self.rough_r_mean,
			"rough_r_stdev": self.rough_r_stdev
		}

	def pickle_data(self):
		file = open(self.PICKLE_FILE_PATH, 'wb')
		pkl.dump(self.data, file)
		file.close()

	def unpickle_data(self):
		file = open(self.PICKLE_FILE_PATH, "rb")
		self.data = pkl.load(file)

	def process_data(self, filename):
		print(str(self.data["smooth_r_stdev"]) + " VS " + str(self.data["rough_r_stdev"]))
		data = self.converter.convert("/users/benfalken/desktop/SideProjectSadness/TestSurfaces", filename)
		r_mean = self.process_image(data, image_index=None, label_index=None, is_individual_image=True)
		smooth_likeness = abs(self.data["smooth_r_mean"] - r_mean)/self.data["smooth_r_stdev"]
		rough_likeness = abs(self.data["rough_r_mean"] - r_mean)/self.data["rough_r_stdev"]
		print("")
		print("Likeness to smooth surface: " + str(smooth_likeness))
		print("Likeness to rough surface: " + str(rough_likeness))
		if (smooth_likeness < rough_likeness):
			print("Likely Smooth")
		else:
			print("Likely Rough")
		print("")
		print("***")