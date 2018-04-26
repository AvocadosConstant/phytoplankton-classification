import cv2
import os
import sys
import numpy as np
from bidict import bidict as bd

#open lst file specified
fp_lst = open(str(sys.argv[1]), "r")
lines_lst = fp_lst.readlines()
fp_lst.close()

#open cla file specified
fp_cla = open(str(sys.argv[2]), "r")
lines_cla = fp_cla.readlines()
fp_cla.close()

algae_index_map = {}
algae_name_counter = 4
while(algae_name_counter < len(lines_cla)):
	algae_type = lines_cla[algae_name_counter].replace("\n", "")
	num_algae = lines_cla[algae_name_counter+3].replace("\n", "")
	
	#create list of indexes associated with particular algae_type
	ind = []
	for x in range(0, int(num_algae)):
		ind.append(lines_cla[algae_name_counter+4+x].replace("\n", ""))

	#create bidict of algae_type and list of indexes
	algae_index_map[algae_type] = ind

	#increase counter to go through loop
	algae_name_counter += int(num_algae)+4


num_field = lines_lst[1].split("|")
num_field = int(num_field[1])

#retrieve information of indecies of where the picture data is, like the x y width and height indexes to be sued in image parsing
for x in range(2, num_field+2):
	if lines_lst[x] == "id|int32\n":
		id_index = x-2
	#	print ("id_index = " + str(id_index))
	elif lines_lst[x] == "image_x|int32\n":
		x_index = x-2
	#	print ("x_index = " + str(x_index))
	elif lines_lst[x] == "image_y|int32\n":
		y_index = x-2
	#	print ("y_index = " + str(y_index))
	elif lines_lst[x] == "image_w|int32\n":
		w_index = x-2
	#	print ("w_index = " + str(w_index))
	elif lines_lst[x] == "image_h|int32\n":
		h_index = x-2
	#	print ("h_index = " + str(h_index))
	elif lines_lst[x] == "collage_file|string\n":
		filename_index = x-2
	#	print ("filename_index = " + str(filename_index))

#current working directory
cwd = os.getcwd()

path_data = "".join([cwd, "/../data/416_Station40_09012015_10x/"])
path_extracted_images = ""
if not os.path.exists("".join([path_data, "extracted_images"])):
	os.makedirs("".join([path_data, "extracted_images"]))
	path_extracted_images += ("".join([path_data, "extracted_images/"]))

#image parsing start at line 66 of file 
for x in range(66, len(lines_lst)):
	des = lines_lst[x].split("|")
	#retrieve images from the data folder above this directory
	src = cv2.imread("".join([path_data,des[filename_index]]))

	x_st = int(des[x_index])
	x_end = int(des[w_index]) + x_st	
	y_st = int(des[y_index])
	y_end = int(des[h_index]) + y_st	
	
	#select the ROI based on X, Y value and width and height
	roi = src[y_st:y_end, x_st:x_end]

	counter = 0
	#create specific folders for each type of algae in extracted_images folder
	for x in range (0, len(list(algae_index_map.values()))):
		ids = list(list(algae_index_map.values())[counter])
		for y in range(0, len(ids)):
			if des[id_index] in ids:
				break
		else:
			counter += 1
			continue
		break
			
	algae_name = list(algae_index_map.keys())[counter]
	path_algae = "".join([path_extracted_images, algae_name, "/"])
	if not os.path.exists(path_algae):
		os.makedirs(path_algae)
	
	# store in extracted_images folder in current directory
	crop_img = "".join( [path_algae, des[id_index], ".tif"])
	cv2.imwrite(crop_img, roi)
#	cv2.waitKey(0)

