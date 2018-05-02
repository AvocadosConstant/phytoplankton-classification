import cv2
import os
import sys
import numpy as np
import glob
from bidict import bidict as bd

def get_directories(data_path):
    directories = glob.glob(data_path)
    return directories
def extractor(cla_path, lst_path,output_dir, input_dir):
	#open lst file specified
	# fp_lst = open(str(sys.argv[1]), "r")
    fp_lst = open(lst_path,"r")
    lines_lst = fp_lst.readlines()
    fp_lst.close()

	#open cla file specified
	# fp_cla = open(str(sys.argv[2]), "r")
    fp_cla = open(cla_path, "r")
    lines_cla = fp_cla.readlines()
    fp_cla.close()
    algae_index_map = {}
    algae_name_counter = 4
    while(algae_name_counter < len(lines_cla)):
	#	print("algae_name_counter " + str(algae_name_counter))
        algae_type = lines_cla[algae_name_counter].replace("\n", "")
	#	print ("algae_type " + algae_type)
        num_algae = lines_cla[algae_name_counter+3].replace("\n", "")
	#	print ("num_algae " + num_algae)

		#create list of indexes associated with particular algae_type
        ind = []
        for x in range(0, int(num_algae)):
            ind.append(lines_cla[algae_name_counter+4+x].replace("\n", ""))
	#	print(ind)

		#create bidict of algae_type and list of indexes
        algae_index_map[algae_type] = ind

		#increase counter to go through loop
        algae_name_counter += int(num_algae)+4
	#	print ("algae_name_counter " + str(algae_name_counter))
    print(algae_index_map)
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

	#Change path data to dynamically get
	#path_data = "".join([cwd, "/../data/416_Station40_09012015_10x/"])
	#print(path_data)
	#os.chdir(path_data)
	#print(os.getcwd())
	#path_extracted_images = ""
	#if not os.path.exists("".join([path_data, "extracted_images"])):
		#os.makedirs("".join([path_data, "extracted_images"]))
		#path_extracted_images += ("".join([path_data, "extracted_images/"]))
	#	print(path_extracted_images)
	#print(path_extracted_images)
    path_data = input_dir
    path_extracted_images=""
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    path_extracted_images=output_dir+"/"


	#image parsing start at line 66 of file
    for x in range(66, len(lines_lst)):
        des = lines_lst[x].split("|")
	#	print(des)
		#retrieve images from the data folder above this directory
        src = cv2.imread("".join([path_data,des[filename_index]]))
	#	print("".join([path_data, des[filename_index]]))

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

	#	print (counter)
        algae_name = list(algae_index_map.keys())[counter]
        path_algae = "".join([path_extracted_images, algae_name, "/"])
        print(path_algae)
        if not os.path.exists(path_algae):
            os.makedirs(path_algae)

		# store in extracted_images folder in current directory
        crop_img = "".join( [path_algae, des[id_index], ".tif"])
        #cv2.imwrite(crop_img, roi)
	#	cv2.waitKey(0)

def main():
    os.chdir('..')
    current = os.getcwd()
    path = os.path.join(current,"data/Multi_Directory_Extraction/*/")
    directories = get_directories(path)
    output_dir = os.path.join(current,"data/training_data")
    print(output_dir)
    for directory_paths in directories:
        station_name = directory_paths.split("/")[-2]
        if(station_name[-2] == '0'):
            lst_path = directory_paths+station_name+".lst"
            cla_path = directory_paths+station_name+".cla"
            print(lst_path)
            print(cla_path)
            extractor(cla_path,lst_path, output_dir,directory_paths)


main()
