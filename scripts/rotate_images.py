import cv2
import sys
import numpy as np
import glob
import os
from PIL import Image


def rotate_bound(image, angle):
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)
	
	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0,0])
	sin = np.abs(M[0,1])
	
	#new bounding dimensions of image
	nW = int((h*sin) + (w*cos))
	nH = int((h*cos) + (w*sin))

	#adjust rotation matrix to take into account translation
	M[0,2] += (nW/2) - cX
	M[1,2] += (nH/2) - cY

	return cv2.warpAffine(image, M, (nW, nH))

cwd = os.getcwd()

#path to extracted images folder
path_to_ex_images = "".join([cwd,"/../data/416_Station40_09012015_10x/extracted_images/"])

#list of algae names
algae_names = ["Aulocoseira", "Asterionella", "Colonial Cyanobacteria", "Cryptomonas", "Detritus", "Dolichospermum", "Filamentous cyanobacteria", "Romeria", "Staurastrum", "Unidentified"]

for algae in algae_names:
	specified_algae_path = "".join([path_to_ex_images, algae, "/*.tif"])
	for filename in glob.glob(specified_algae_path):
		original_image = cv2.imread(filename)
		rotated_image = rotate_bound(original_image, 90)
		cv2.imshow(filename, rotated_image)
		cv2.waitKey(0)
		break;
	break;
