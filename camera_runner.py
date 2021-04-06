#from picamera import PiCamera
import time

class CameraRunner:
	def __init__(self, model):
		self.model = model
		self.lower_image_num = 10
		self.higher_image_num = 16
		self.delay = 2
		#self.PATH = '/users/benfalken/desktop/SideProjectSadness/TestSurfaces/'
		#self.camera = PiCamera()
		#self.camera.resolution = (1024, 768)
	def run(self):
		#self.camera.start_preview()
		for i in range(self.lower_image_num, self.higher_image_num):
			image_name = "images-" + str(i+1) + ".jpeg"
			#self.camera.capture(image_name)
			self.model.process_data(image_name)
			time.sleep(self.delay)