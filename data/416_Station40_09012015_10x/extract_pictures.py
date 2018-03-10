import cv2
import sys
import numpy as np

#print file name
#print (str(sys.argv[1]))

#open file specified
fp = open(str(sys.argv[1]), "r")

lines = fp.readlines()

fp.close()

num_field = lines[1].split("|")
num_field = int(num_field[1])

#print (num_field)

for x in range(2, num_field+2):
	if lines[x] == "id|int32\n":
		id_index = x-2
	#	print ("id_index = " + str(id_index))
	elif lines[x] == "image_x|int32\n":
		x_index = x-2
	#	print ("x_index = " + str(x_index))
	elif lines[x] == "image_y|int32\n":
		y_index = x-2
	#	print ("y_index = " + str(y_index))
	elif lines[x] == "image_w|int32\n":
		w_index = x-2
	#	print ("w_index = " + str(w_index))
	elif lines[x] == "image_h|int32\n":
		h_index = x-2
	#	print ("h_index = " + str(h_index))
	elif lines[x] == "collage_file|string\n":
		filename_index = x-2
	#	print ("filename_index = " + str(filename_index))



#image parsing start at line 66 of file 
for x in range(66, len(lines)):
	des = lines[x].split("|")
	src = cv2.imread(des[filename_index])

	x_st = int(des[x_index])
	x_end = int(des[w_index]) + x_st	
	y_st = int(des[y_index])
	y_end = int(des[h_index]) + y_st	
	
	#select the ROI based on X, Y value and width and height
	roi = src[y_st:y_end, x_st:x_end]

	crop_img = "".join( ["extracted_images/",des[id_index], ".tif"])
	cv2.imwrite(crop_img, roi)
#	cv2.imshow("Image", roi)

#	cv2.waitKey(0)

