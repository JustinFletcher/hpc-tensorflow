## astroJsonParser.py
## gmartin @ 02/02/2018
## INPUT:
##      JSON A folder of collections. 
##		Each collection contains a directory of AST files converted to json format.
##		Also contained in each folder is a "Mechanical Turk" file containing the observations of examined files
##      

import os
import sys
from shutil import copyfile
import cv2
import re
import shutil
import collections
import json
import argparse
import numpy as np
import math
from random import randint
from jsonParserTurk import Turk
from jsonParserAstro import Astro
from collections import defaultdict
from collections import Counter
from operator import itemgetter

SCALE = [2, 4, 8]
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

REGION_BOUND = {2:256,4:128,8:64}
# NUM_REGIONS = {2:4,4:16,8:64}

str_csv_separator = ' '
str_new_line = '\n'

## CHECK INPUT DIRECTORY
def functCheckDir(inputDir):
    if not os.path.isdir(inputDir):
        print("\t\n ## Warning: Unable to find directory ##\n")
        sys.exit()
    else:
        return inputDir

## SELECT SPECIFIC FILE
def functGetFiles(inputDir, inExtension):
    file_List = []
    for mainDir, subDir, fileName in os.walk(inputDir):
        for listFiles in os.listdir(mainDir):
            if listFiles.endswith(inExtension):
                file_List.append(os.path.normpath(os.path.join(mainDir, listFiles)))
        return file_List

## PARSE ASTROMETRY
def functExtractAstrometry(inputFile, inputString):
    temp_List = []
    with open(inputFile) as fileInput:
        for textContent in fileInput:
            if inputString in textContent:
                for listDetection in fileInput:
                    temp_List.append(listDetection.rstrip())
    objDetection = filter(None, temp_List)
    return objDetection
    
## WRITE TO FILE 
def functWriteToFile(inFile, inContent):
    fileWrite = open(inFile, "a")
    fileWrite.write(inContent + "\n")

    fileWrite.close()

def read_annotation_file(filename):
    """Reads an annotation file.

    Converts an astronet annotation file into a dictionary containing all the
    relevant information.

    Args:
    filename: the path to the annotataion text file.

    Returns:
    anno: A dictionary with the converted annotation information. 
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]

    # fix for label: index is off by 1
    # label = [int(x[0])+1 for x in content]
    label = [1 for x in content]
    # print('label:',label)

    x_center = [float(x[1]) for x in content]
    y_center = [float(x[2]) for x in content]
    bbox_width = [float(x[3]) for x in content]
    bbox_height = [float(x[4]) for x in content]

    # clip ranges to 0 and 1
    y_min = [max(y0-h/2.,0.) for y0, h in zip(y_center, bbox_height)]
    y_max = [min(y0+h/2.,1.) for y0, h in zip(y_center, bbox_height)]

    x_min = [max(x0-w/2.,0.) for x0, w in zip(x_center, bbox_width)]
    x_max = [min(x0+w/2.,1.) for x0, w in zip(x_center, bbox_width)]

    # adjust bbox_width and bbox_height for those x0,y0 values that have been clipped
    bbox_height = [2.*y0 if ymn == 0. else h for ymn,y0,h in zip(y_min, y_center, bbox_height)]
    bbox_height = [2.*(1.-y0)  if ymx == 1. else h for ymx,y0,h in zip(y_max, y_center, bbox_height)]

    bbox_width = [2*x0 if xmn == 0. else w for xmn,x0,w in zip(x_min, x_center, bbox_width)]
    bbox_width = [2.*(1.-x0) if xmx == 1. else w for xmx,x0,w in zip(x_max, x_center, bbox_width)]

    # bbox_width and bbox_height should be symmetric, i.e. square: make them so:
    bbox_width = [h if h < w else w for w,h in zip(bbox_width, bbox_height)]
    bbox_height = [w if w < h else h for w,h in zip(bbox_width, bbox_height)]

    anno = {}
    anno['label'] = label

    anno['x_center'] = x_center
    anno['y_center'] = y_center

    anno['bbox_width'] = bbox_width
    anno['bbox_height'] = bbox_height

    anno['y_min'] = y_min
    anno['y_max'] = y_max

    anno['x_min'] = x_min
    anno['x_max'] = x_max
    
    anno['min_y_min'] = min(y_min)
    anno['max_y_max'] = max(y_max)

    anno['min_x_min'] = min(x_min)
    anno['max_x_max'] = max(x_max)

    anno['mean_x_center'] = np.mean(x_center)
    anno['mean_y_center'] = np.mean(y_center)

    # print('anno:',anno)
    return anno

def can_scale(annot_dict, scale):
	scale = 1.0/scale
	if ((annot_dict['max_x_max'] - annot_dict['min_x_min']) < scale) and ((annot_dict['max_y_max'] - annot_dict['min_y_min']) < scale):
		return True
	else:
		return False

# find which ROI the objects are located in.
# Note: can_scale since already determined that the object(s) can fit in an ROI.
# Scale = 2X, 4X, 8X
# ROIs  =  4, 16, 64
# def  get_roi(annot_dict, scale):

# 	bound = REGION_BOUND[scale]
# 	# n_regions = NUM_REGIONS[scale]

# 	x0 = math.ceil(annot_dict['mean_x_center']*IMAGE_WIDTH)
# 	y0 = math.ceil(annot_dict['mean_y_center']*IMAGE_HEIGHT)

# 	min_y_min = math.ceil(annot_dict['min_y_min']*IMAGE_WIDTH)
# 	max_y_max = math.ceil(annot_dict['max_y_max']*IMAGE_WIDTH)

# 	min_x_min = math.ceil(annot_dict['min_x_min']*IMAGE_WIDTH)
# 	max_x_max = math.ceil(annot_dict['max_x_max']*IMAGE_WIDTH)

# 	# check boundaries and prevent clipping:
# 	x_min = 0
# 	y_min = 0

# 	# save corrections for updating scaled annotations:
# 	x0_cor = 0.
# 	y0_cor = 0.

# 	offset = int(bound/2)

# 	x_max = (IMAGE_WIDTH-offset)
# 	y_max = (IMAGE_HEIGHT-offset)

# 	# adjust for center of satellites:
# 	if x0 > x_max:
# 		x0_cor = (x_max-x0)/IMAGE_WIDTH
# 		x0 = x_max

# 	if y0 > y_max:
# 		y0_cor = (y_max-y0)/IMAGE_HEIGHT
# 		y0 = y_max

# 	if (x0-offset) < x_min:
# 		x0_cor = -(x0-offset)/IMAGE_WIDTH
# 		x0 = x_min + offset

# 	if (y0-offset) < y_min:
# 		y0_cor = -(y0-offset)/IMAGE_HEIGHT
# 		y0 = y_min + offset

# 	# adjust for extrema:
# 	delX = max_x_max - x0
# 	if delX > offset:
# 		x0_cor += (delX-offset)/IMAGE_WIDTH
# 		x0 += delX-offset

# 	delY = max_y_max - y0
# 	if delY > offset:
# 		y0_cor += (delY-offset)/IMAGE_HEIGHT
# 		y0 += delY-offset

# 	delX = -(min_x_min - x0)
# 	if delX > offset:
# 		x0_cor -= (delX-offset)/IMAGE_WIDTH
# 		x0 -= delX-offset

# 	delY = -(min_y_min - y0)
# 	if delY > offset:
# 		y0_cor -= (delY-offset)/IMAGE_HEIGHT
# 		y0 -= delY-offset

# 	roi = {}
# 	roi['x'] = x0
# 	roi['y'] = y0
# 	roi['w'] = bound
# 	roi['h'] = bound
# 	roi['x0_cor'] = x0_cor
# 	roi['y0_cor'] = y0_cor

# 	print('x0,max_x_max,x_max,offset:',x0,max_x_max,x_max,offset)
# 	print('get_roi:',roi)
# 	return roi

def get_roi2(annot_dict, scale):

	bound = REGION_BOUND[scale]
	roi_range_dict = get_roi_range(annot_dict, scale)
	# print('roi_range_dict:')
	# print(roi_range_dict)

	xc_min = roi_range_dict['xc_min']
	xc_max = roi_range_dict['xc_max']

	yc_min = roi_range_dict['yc_min']
	yc_max = roi_range_dict['yc_max']

	x_c = randint(xc_min, xc_max)
	y_c = randint(yc_min, yc_max)

	roi = {}
	roi['x'] = x_c
	roi['y'] = y_c
	roi['w'] = bound
	roi['h'] = bound
	roi['x0_cor'] = x_c/IMAGE_WIDTH
	roi['y0_cor'] = y_c/IMAGE_HEIGHT

	# print('get_roi2:',roi)
	return roi

def get_roi_range(annot_dict, scale):

	# get the min and max (x,y) center coordinates of the ROI region for extraction:
	# (x_c_min, x_c_max) and (y_c_min, y_c_max)

	# Image dimensions:
	Wi = IMAGE_WIDTH
	Hi = IMAGE_HEIGHT

	# ROI of extraction window:
	Wr = REGION_BOUND[scale]
	Hr = REGION_BOUND[scale]

	# Ground truth bounded box height and width:
	bbox_width = [math.ceil(bbw*IMAGE_WIDTH) for bbw in annot_dict['bbox_width']]
	bbox_height = [math.ceil(bbh*IMAGE_HEIGHT) for bbh in annot_dict['bbox_height']]

	x_center = [math.ceil(xc*IMAGE_WIDTH) for xc in annot_dict['x_center']]
	y_center = [math.ceil(yc*IMAGE_HEIGHT) for yc in annot_dict['y_center']]

	xc_min_l = []
	xc_max_l = []
	yc_min_l = []
	yc_max_l = []

	for x0,y0,Wb,Hb in zip(x_center,y_center,bbox_width,bbox_height):
		# determine min and max possible values for xc, yc
		xc_min = x0 + Wb/2 - Wr/2
		xc_max = x0 - Wb/2 + Wr/2

		yc_min = y0 + Hb/2 - Hr/2
		yc_max = y0 - Hb/2 + Hr/2

		if (xc_min - Wr/2) < 0:
			xc_min = Wr/2

		if (yc_min - Hr/2) < 0:
			yc_min = Hr/2

		if (xc_max + Wr/2) > Wi:
			xc_max = Wi - Wr/2

		if (yc_max + Hr/2) > Hi:
			yc_max = Hi - Hr/2

		xc_min_l.append(xc_min)
		xc_max_l.append(xc_max)
		yc_min_l.append(yc_min)
		yc_max_l.append(yc_max)

	# print('xc_min_l:',xc_min_l)
	# print('xc_max_l:',xc_max_l)	

	roi_range_dict = {}
	roi_range_dict['xc_min'] = math.ceil(max(xc_min_l))
	roi_range_dict['xc_max'] = math.ceil(min(xc_max_l))

	roi_range_dict['yc_min'] = math.ceil(max(yc_min_l))
	roi_range_dict['yc_max'] = math.ceil(min(yc_max_l))

	# range check:
	if roi_range_dict['xc_min'] > roi_range_dict['xc_max']:
		roi_range_dict['xc_min'] = roi_range_dict['xc_max']

	if roi_range_dict['yc_min'] > roi_range_dict['yc_max']:
		roi_range_dict['yc_min'] = roi_range_dict['yc_max']

	return roi_range_dict

def scale_roi(roi,s):
	roi['x'] = roi['x']*s
	roi['y'] = roi['y']*s
	roi['w'] = roi['w']*s
	roi['h'] = roi['h']*s
	return roi

# def scale_annot(annot_dict, scale,image_roi_dict):

# 	bound = REGION_BOUND[scale]
# 	# n_regions = NUM_REGIONS[scale]

# 	# x_region = int(annot_dict['mean_x_center']*IMAGE_WIDTH/bound)
# 	# y_region = int(annot_dict['mean_y_center']*IMAGE_HEIGHT/bound)

# 	x0 = annot_dict['mean_x_center']
# 	y0 = annot_dict['mean_y_center']

# 	# corrections applied to extracting image ROI:
# 	x0_cor = image_roi_dict['x0_cor']
# 	y0_cor = image_roi_dict['y0_cor']

# 	# centroid of satellite coords is centered 
# 	offset = 0.5

# 	roi = {}

# 	# offset_x = float(x_region)*bound/IMAGE_WIDTH
# 	# offset_y = float(y_region)*bound/IMAGE_HEIGHT

# 	roi['label'] = annot_dict['label']

# 	roi['x_center'] = [(x-x0_cor-x0)*scale+offset for x in annot_dict['x_center']]
# 	roi['y_center'] = [(y-y0_cor-y0)*scale+offset for y in annot_dict['y_center']]

# 	roi['bbox_width'] = [w*scale for w in annot_dict['bbox_width']]
# 	roi['bbox_height'] = [h*scale for h in annot_dict['bbox_height']]

# 	# print('***')
# 	# print('scale_annot:',scale)
# 	# print('x_region:', x_region)
# 	# print('y_region:', y_region)
# 	# print('offset_x:', offset_x)
# 	# print('offset_y:', offset_y)
# 	# print('roi[x_center]:', roi['x_center'])
# 	# print('roi[y_center]:', roi['y_center'])
# 	# print('annot_dict[]:', annot_dict)
# 	# print('')

# 	return roi

def scale_annot2(annot_dict, scale,image_roi_dict):

	bound = REGION_BOUND[scale]
	# n_regions = NUM_REGIONS[scale]

	# x_region = int(annot_dict['mean_x_center']*IMAGE_WIDTH/bound)
	# y_region = int(annot_dict['mean_y_center']*IMAGE_HEIGHT/bound)

	# x0 = annot_dict['mean_x_center']
	# y0 = annot_dict['mean_y_center']

	# corrections applied to extracting image ROI:
	x0_cor = image_roi_dict['x0_cor']
	y0_cor = image_roi_dict['y0_cor']

	# centroid of satellite coords is centered 
	offset = 0.5

	roi = {}

	# offset_x = float(x_region)*bound/IMAGE_WIDTH
	# offset_y = float(y_region)*bound/IMAGE_HEIGHT

	roi['label'] = annot_dict['label']

	roi['x_center'] = [(x-x0_cor)*scale+offset for x in annot_dict['x_center']]
	roi['y_center'] = [(y-y0_cor)*scale+offset for y in annot_dict['y_center']]

	roi['bbox_width'] = [w*scale for w in annot_dict['bbox_width']]
	roi['bbox_height'] = [h*scale for h in annot_dict['bbox_height']]

	# print('***')
	# print('scale_annot:',scale)
	# print('x_region:', x_region)
	# print('y_region:', y_region)
	# print('offset_x:', offset_x)
	# print('offset_y:', offset_y)
	# print('roi[x_center]:', roi['x_center'])
	# print('roi[y_center]:', roi['y_center'])
	# print('annot_dict[]:', annot_dict)
	# print('')

	return roi

def scale_annot_2_txt(roi_dict):
	str_out_total = ''
	# str_csv_separator = ','
	# str_new_line = '\n'
	label = roi_dict['label']
	x0 = roi_dict['x_center']
	y0 = roi_dict['y_center']
	w = roi_dict['bbox_width']
	h = roi_dict['bbox_height']
	n = len(x0)
	# generate an output string composed of: class_index x0 y0 w h
	for i in range(n):
		class_index_str = '0'
		x0_str = '{:6.4f}'.format(x0[i])
		y0_str = '{:6.4f}'.format(y0[i])
		w_str = '{:6.4f}'.format(w[i])
		h_str = '{:6.4f}'.format(h[i])
		str_out =  class_index_str + str_csv_separator + x0_str + str_csv_separator + y0_str + str_csv_separator + w_str + str_csv_separator + h_str + str_new_line
		# print('str_out:',str_out)
		str_out_total = str_out_total + str_out
	return str_out_total

## MAIN FUNCTION
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d","--dir", type=str, default='../data_uct_enabled/JSON', help="path to JSON data dir")
	parser.add_argument("-p","--parse", type=str, help="parse Astro or Turk data")
	parser.add_argument("-a","--annot", action="store_true", help="generate annotation data")
	parser.add_argument("-t","--turk2annot", action="store_true", help="update annotation data with human turk data")
	parser.add_argument("-k","--darknetTrain", action="store_true", help="generate data for training via darknet YOLO.v2 model")
	parser.add_argument("-s","--scale", action="store_true", help="scale up satellite imagery by 2X, 4X and 8X.")
	parser.add_argument('--directory', type=str, default='~/tensorflow/false_positives/data_uct_enabled', help='image_data_directory')
	parser.add_argument("--stats", action="store_true", help="generate statistics on number of satellite detections per image")
	parser.add_argument("--compare", action="store_true", help="compare two datasets for number of similar detections")
	parser.add_argument('--directory2', type=str, default='../datat/JSON', help='directory for comparison')
	# totalArg = len(sys.argv)
	# if(totalArg != 3):
	# 	print("\n +++  Verify Number of Arguments +++ \n")
	# 	print("Command Format: python astroJsonParser.py <Json Folder > <parseTurk or parseAstro\n")
	# 	sys.exit()
	args = parser.parse_args()
	#jsonDir = str(sys.argv[1])
	jsonDir = args.dir

	# datan = False
	# if 'datan' in jsonDir:
	# 	datan = True
	# datan = True

	#whichParser = str(sys.argv[2])
	whichParser = args.parse

	print("jsonDir:", jsonDir)
	print("whichParser",whichParser)

	box_width = 20
	box_height = 20

	class_name = 'sat'
	class_index = 0
	class_index_str = '0'

	image_width = 512
	image_height = 512

	str_box_width = str(box_width)
	str_box_height = str(box_height)

	# str_csv_separator = ' '
	# str_new_line = '\n'

if(whichParser == 'Turk'):
	# parseTurk
	print('parseTurk')
	#sys.exit()

	turk_data = Turk()

	getSourceDir = functCheckDir(jsonDir)
	#print('sourceDir:', getSourceDir)

	onlyDirs = sorted([f for f in os.listdir(getSourceDir) if os.path.isdir(os.path.join(getSourceDir, f))])

	#print('dirs:', onlyDirs)

	# for each directory, read the "Mechanical Turk" file and the directory of AST files converted to Json:
	for f in onlyDirs:
		mechTurkFile = functGetFiles(os.path.join(getSourceDir, f),'.json')
		print('mechTurkFile:',mechTurkFile)
		mechTurkFile = mechTurkFile[0] # it's a list of files                 
		#print('turkFile:', mechTurkFile) # it's a list of files
		# parse the turk file and save results to array of lists:
		#turk_data = Turk(mechTurkFile[0])
		turk_data.addTurkFile(mechTurkFile)

	# save data to pickle file:
	print("\nBefore save...\n")
	turk_data.info()

	pickleFile = os.path.join(jsonDir, 'mechanicalTurk.pickle')
	print(pickleFile)
	turk_data.saveTurkDataAsPickle(pickleFile)

	print('\nAfter save...\n')
	turk_data.info()

elif(whichParser == 'Astro'):
	#parseAstro
	print('parseAstro')
	#sys.exit()

	astro_data = Astro()

	getSourceDir = functCheckDir(jsonDir)
	#print('sourceDir:', getSourceDir)

	onlyDirs = sorted([f for f in os.listdir(getSourceDir) if os.path.isdir(os.path.join(getSourceDir, f))])

	#print('dirs:', onlyDirs)

	# for each directory, read the AST files in the directory of AST files converted to Json:
	for f in onlyDirs:
		path = os.path.join(getSourceDir, f)
		#print(path)
		# for f2 in os.listdir(path): 
		f2_json = [x for x in os.listdir(path) if 'astfiles_to_json' in x ] 
		print(f2_json[0])
		# sys.exit()
		if os.path.isdir(os.path.join(path, f2_json[0])):
			path2 = os.path.join(path, f2_json[0])
			#print(path2)
			#get the list of astro files:
			astro_file_list = functGetFiles(path2, 'json')
			#print(astro_file_list[0])
			astro_data.addAstroFile(astro_file_list)


	pickleFile = os.path.join(jsonDir, 'astroData.pickle')
	print(pickleFile)
	astro_data.saveAstroDataAsPickle(pickleFile)

	print('\nAfter save...\n')
	astro_data.info('All',True)

elif(args.annot == True):
	print("generate annotation data:...")

	# box_width = 20
	# box_height = 20
	# class_name = 'sat'

	# str_box_width = str(box_width)
	# str_box_height = str(box_height)

	# str_csv_separator = ', '

	astro_data = Astro()
	
	# get the direrctories in which the FITS directories are contained:
	sourceDir =  os.path.abspath(jsonDir)
	print('sourceDir:', sourceDir)

	# load the parsed astro data:
	pickle_file = os.path.join(sourceDir, 'astroData.pickle')
	astro_data.loadAstroDataFromPickle(pickle_file)

	# save output path
	sinkDir = sourceDir

	#fitsDir = os.chdir('..')
	sourceDir = sourceDir.replace('/JSON','')
	print('fitsDir:', sourceDir)

	fitsDirs = sorted([f for f in os.listdir(sourceDir) if os.path.isdir(os.path.join(sourceDir, f))])
	print('fitsDir:', fitsDirs)

	fitsDirs = [ x for x in fitsDirs if "." in x ]
	print('fitsDirs:', fitsDirs)

	#sys.exit()

	# for each directory, read the AST files in the directory of AST files converted to Json:
	for d in fitsDirs:
		pathDir = os.path.join(sourceDir, d,'FITS')

		# create the output directory
		d_out = d.split('.')
		# d_dir_out = d_out[0] + '.' + d_out[2] + '.' + d_out[3] + '.' + d_out[1]
		# if datan:
		d_dir_out = d_out[0] + '.' + d_out[2] + '.' + d_out[3] + '.' + d_out[1]
		# else:
		# 	d_dir_out = d_out[2] + '.' + d_out[3] + '.' + d_out[1]

		dir_out = os.path.join(sinkDir, d_dir_out,'ImageFiles')
		if not os.path.exists(dir_out):
			os.makedirs(dir_out)

		dir_out_ann = os.path.join(sinkDir, d_dir_out,'ImageFilesAnnotated')
		if not os.path.exists(dir_out_ann):
			os.makedirs(dir_out_ann)

		annot_dir_out = os.path.join(sinkDir, d_dir_out,'Annotations')
		if not os.path.exists(annot_dir_out):
			os.makedirs(annot_dir_out)

		print(pathDir)

		fits_file_list = functGetFiles(pathDir, 'fits')
		nFiles = len(fits_file_list)
		# print('fits_file_list:')
		# print('nFiles:', nFiles)
		# print(fits_file_list[0])
		# print(fits_file_list[1])
		# print(fits_file_list[2])
		# print(fits_file_list[nFiles-1])

		for f in fits_file_list:
			#print('reading {}'.format(f))
			img_equal_rgb = astro_data.read_fits(f)
			# fits_data = astro_data.read_fits(f)
			# img_equal_rgb = astro_data.histogram_equalize(fits_data)
			# img_rgb = cv2.cvtColor(fits_data, cv2.COLOR_GRAY2RGB)
			# img_rgb = fits_data
			img_rgb = img_equal_rgb

			# get the file name:
			fileNameFits = os.path.basename(f)
			fileNameJpg = fileNameFits.replace('fits','jpg')
			path2Jpg = os.path.join(dir_out,fileNameJpg)

			# img_rgb = cv2.convertScaleAbs(img_rgb, alpha=(255.0/65535.0))

			# print('writing {}'.format(path2Png))
			# print('before save as png:')
			# print('img size:', img_rgb.size)
			# print('img type:', img_rgb.dtype)
			# print('img bytes:', img_rgb.nbytes)

			astro_data.save_image2(img_rgb,path2Jpg )
			# cv2.imwrite(img_rgb, path2Png)


			# img_rgb = cv2.imread(path2Png,cv2.IMREAD_COLOR)
			# img_rgb = cv2.imread(path2Png,cv2.IMREAD_COLOR)
			# img_rgb = cv2.imread(path2Png,cv2.CV_LOAD_IMAGE_ANYDEPTH)

			# print('writing {}'.format(path2Png))
			# print('after save as png:')
			# print('img size:', img_rgb.size)
			# print('img type:', img_rgb.dtype)
			# print('img bytes:', img_rgb.nbytes)

			# sys.exit()

			# get the list of (x,y) coordinates for this FITS image:
			xy_list = astro_data.getXYCoordsForFITS(d_dir_out,fileNameFits)
			# save coordinates in an annotation file:
			fileNameAnnot = fileNameFits.replace('fits','txt')
			path2Annot = os.path.join(annot_dir_out,fileNameAnnot)
			f_annot = open(path2Annot,'w')
			for t in xy_list:
				x_str = '{:7.3f}'.format(t[0])
				y_str = '{:7.3f}'.format(t[1])
				str_out =  d_dir_out + str_csv_separator + fileNameFits + str_csv_separator + class_index_str + str_csv_separator + x_str + str_csv_separator + y_str + str_csv_separator + str_box_width + str_csv_separator + str_box_height + str_new_line
				f_annot.write(str_out)
			f_annot.close()
			# print(d_dir_out, fileNameFits, xy_list)

			# save ground truth annotated images
			path2JpgAnnot = os.path.join(dir_out_ann,fileNameJpg)
			img_annot_rgb = astro_data.annotateImage(img_equal_rgb,class_name,xy_list,box_width,box_height)
			astro_data.save_image2(img_annot_rgb,path2JpgAnnot)

			# print('Done')
			# sys.exit()

	print('Done')

elif(args.turk2annot == True):
	print("update annotations with Human Turk data:...")

	# box_width = 20
	# box_height = 20
	# class_name = 'sat'

	# str_box_width = str(box_width)
	# str_box_height = str(box_height)

	# str_csv_separator = ', '

	turk_data = Turk()
	astro_data = Astro()
	
	# get the direrctories in which the FITS directories are contained:
	sourceDir =  os.path.abspath(jsonDir)
	print('sourceDir:', sourceDir)

	# load the parsed astro data:
	pickle_file = os.path.join(sourceDir, 'mechanicalTurk.pickle')
	turk_data.loadTurkDataFromPickle(pickle_file)

	getSourceDir = functCheckDir(jsonDir)
	#print('sourceDir:', getSourceDir)

	onlyDirs = sorted([f for f in os.listdir(getSourceDir) if os.path.isdir(os.path.join(getSourceDir, f))])

	#print('dirs:', onlyDirs)

	# for each directory, read the "Mechanical Turk" file and the directory of AST files converted to Json:
	for f in onlyDirs:
		mechTurkFile = functGetFiles(os.path.join(getSourceDir, f),'json')
		mechTurkFile = mechTurkFile[0] # it's a list of files
		#print('turkFile:', mechTurkFile) # it's a list of files
		# parse the turk file and save results to array of lists:
		#turk_data = Turk(mechTurkFile[0])
		turk_file_name = mechTurkFile.split('/')[3]

		# find all the false negative detections from the Turk data:
		false_negatives_dict = turk_data.getFalseNegatives(turk_file_name)

		print(turk_file_name)
		print('false negatives:')
		print(len(false_negatives_dict['imagefilename']))
		print(len(false_negatives_dict['xcenter']))
		print(len(false_negatives_dict['ycenter']))

		if(len(false_negatives_dict['imagefilename'])>0):
			print(false_negatives_dict['imagefilename'][0])
			print(false_negatives_dict['xcenter'][0])
			print(false_negatives_dict['ycenter'][0])
		print()

		# find all the false positive detections from the Turk data:
		false_positives_dict = turk_data.getFalsePositives(turk_file_name)

		print('false positives:')
		print(len(false_positives_dict['imagefilename']))
		print(len(false_positives_dict['xcenter']))
		print(len(false_positives_dict['ycenter']))

		if(len(false_positives_dict['imagefilename'])>0):
			print(false_positives_dict['imagefilename'][0])
			print(false_positives_dict['xcenter'][0])
			print(false_positives_dict['ycenter'][0])
		print()
		print()

		# add the false negative detections to the annotation data and update the Annotaed Images:
		for i,v in enumerate(false_negatives_dict['imagefilename']):
			fileNameFits = v
			x_center = false_negatives_dict['xcenter'][i]
			y_center = false_negatives_dict['ycenter'][i]
			path2Annot = os.path.join(sourceDir, f, 'Annotations', fileNameFits.replace('fits','txt'))
			f_annot = open(path2Annot,'a')
			x_str = '{:7.3f}'.format(x_center)
			y_str = '{:7.3f}'.format(y_center)
			str_out =  turk_file_name + str_csv_separator + fileNameFits + str_csv_separator + class_index_str + str_csv_separator + x_str + str_csv_separator + y_str + str_csv_separator + str_box_width + str_csv_separator + str_box_height + str_new_line
			f_annot.write(str_out)
			f_annot.close()

			path2JpgAnnot = os.path.join(sourceDir, f,'ImageFilesAnnotated', fileNameFits.replace('fits','jpg'))
			astro_data.addtoAnnotatedImage(path2JpgAnnot,class_name,x_center,y_center,box_width,box_height)

		# remove the false positive detections from annotation data and update the Annotated Images:
		for i,v in enumerate(false_positives_dict['imagefilename']):
			fileNameFits = v
			x_center = false_positives_dict['xcenter'][i]
			y_center = false_positives_dict['ycenter'][i]
			path2Annot = os.path.join(sourceDir, f, 'Annotations', fileNameFits.replace('fits','txt'))

			# open the file, read it, find the values to remove, remove it and then write the data to the same file:
			x_str = '{:7.3f}'.format(x_center)
			y_str = '{:7.3f}'.format(y_center)
			xy_list = []

			with open(path2Annot) as fptr:
				file_str = fptr.read()

			# separate text into lines and parse each line that contains x_center, y_center:
			text_list = file_str.split(str_new_line)

			i_remove = None
			for i,l in enumerate(text_list):
				if (x_str in l) and (y_str in l):
					i_remove = i
				else:
					fields = l.split()
					if len(fields) < 6:
						continue
					print('fields:',fields)
					x_i = float(fields[3])
					y_i = float(fields[4])
					xy_list.append((x_i,y_i))

			if i_remove is not None:
				print('removing line:',text_list[i_remove])
				print('before:',text_list)
				del(text_list[i_remove])
				print('after',text_list)

			if text_list:
				print('joining:')
				file_str = '\n'.join(["%s" % (v) for k, v in enumerate(text_list)])
			else:
				print('not joining')
				file_str = ''

			print('file_str:')
			print(file_str)
			# sys.exit()

			with open(path2Annot, "w") as fptr:
				fptr.write(file_str)

			# create a newly annotated image from the non-annoted image:
			path2Jpg = os.path.join(sourceDir, f, 'ImageFiles', fileNameFits.replace('fits','jpg'))
			# read as color
			img_rgb = cv2.imread(path2Jpg,cv2.IMREAD_COLOR)
			# img_gray = cv2.imread(path2Png,cv2.IMREAD_GRAYSCALE)
			# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
			# img_equal_rgb = astro_data.histogram_equalize(img_gray)


			print('xy_list',xy_list)
			path2JpgAnnot = os.path.join(sourceDir, f,'ImageFilesAnnotated', fileNameFits.replace('fits','jpg'))
			img_annot = astro_data.annotateImage(img_rgb,class_name,xy_list,box_width,box_height)
			astro_data.save_image2(img_annot,path2JpgAnnot)
			# sys.exit()

	print('Done')

elif(args.darknetTrain == True):

	print("generate data for training via darknet YOLO.v2 model:...")
	# get the direrctories in which the FITS directories are contained:
	sourceDir =  os.path.abspath(jsonDir)
	# print('sourceDir:', sourceDir)

	path2Output = sourceDir.replace('JSON','image_data')
	# print('path2Output:',path2Output)

	getSourceDir = functCheckDir(jsonDir)
	#print('sourceDir:', getSourceDir)

	onlyDirs = sorted([f for f in os.listdir(getSourceDir) if os.path.isdir(os.path.join(getSourceDir, f))])

	# print('dirs:', onlyDirs)

	# for each directory, read the "Mechanical Turk" file and the directory of AST files converted to Json:
	for f in onlyDirs:
		path2Annot = os.path.join(sourceDir, f, 'Annotations')
		# path2Images = os.path.join(sourceDir, f, 'ImageFilesAnnotated') # change to 'ImageFiles' after validating data visually
		path2Images = os.path.join(sourceDir, f, 'ImageFiles')
		# print('path2Annot:',path2Annot)
		# print('path2Images:',path2Images)
		# get the list of annotation files in the Annotations directory:
		annotationsList = functGetFiles(path2Annot,'txt')
		for annot in annotationsList:
			# print('annot:',annot)
			# open the annotations file and read it:
			str_out_total = ''
			with open(annot) as fptr:
				file_str = fptr.read()
				# separate text into lines and parse each line
				text_list = file_str.split(str_new_line)
				# drop the last element in the list: '':
				text_list = text_list[:-1]
				# print('text_list:',text_list)
				# print('len(text_list):',len(text_list))
				# skip the file if there are no annotations:
				if len(text_list) > 0:
					for s in text_list:
						# get the x_cntr, y_cntr, w, h info:
						fields = s.split()
						# print('s:',s)
						# print('fields:',fields)
						x0 = float(fields[3])
						y0 = float(fields[4])
						w = float(fields[5])
						h = float(fields[6])

						# verify that x0 and y0 are within limits;
						if not ((0 <= x0 <= image_width) and (0 <= y0 <= image_height)):
							print('object location exceeds bounds:', annot)
						# print(x0,y0,w,h)
						# convert to darknet format: relative to the H,W of the image H,W:
						x0 = x0/image_width
						y0 = y0/image_height
						w = w/image_width
						h = h/image_height
						# print(x0,y0,w,h)
						# generate an output string composed of: class_index x0 y0 w h
						x0_str = '{:6.4f}'.format(x0)
						y0_str = '{:6.4f}'.format(y0)
						w_str = '{:6.4f}'.format(w)
						h_str = '{:6.4f}'.format(h)
						str_out =  class_index_str + str_csv_separator + x0_str + str_csv_separator + y0_str + str_csv_separator + w_str + str_csv_separator + h_str + str_new_line
						# print('str_out:',str_out)
						str_out_total = str_out_total + str_out
			# print('str_out_total:')
			# print(str_out_total)
			# write str_out_total to file if not empty:
			if str_out_total != '':
				annotBaseName = os.path.basename(annot)
				fname_out = f + '_' + annotBaseName
				# print('fname_out:',fname_out)
				path2AnnotOut = os.path.join(path2Output,fname_out)
				# print('path2AnnotOut:',path2AnnotOut)
				with open(path2AnnotOut, "w") as fptr:
					fptr.write(str_out_total)
				# copy the image file to the same directory:
				imageBaseName = annotBaseName.replace('txt','jpg')
				fname_out = f + '_' + imageBaseName
				# print('fname_out:',fname_out)
				path2ImageOut = os.path.join(path2Output,fname_out)
				# print('path2ImageOut:',path2ImageOut)
				#copy the file:
				path2ImageSrc = os.path.join(path2Images,imageBaseName)
				# print('path2ImageSrc:',path2ImageSrc)
				copyfile(path2ImageSrc, path2ImageOut)
	# sys.exit()

	print('Done')

elif(args.scale == True):

	print("generate scaled satellite data for training: 1X, 2X, 4X and 8X")

	astro_data = Astro()

	sourceDir = os.path.expanduser(args.directory)

	# image directory - input
	path_to_image_data = os.path.join(sourceDir,'image_data')
	# print('path_to_image_data:',path_to_image_data)

	# scaled image directory - output
	path_to_image_data_s = os.path.join(sourceDir,'image_data_s')

	# ground truth image directory - output
	path_to_image_data_gt = os.path.join(sourceDir,'image_data_gt')

	# get the list of image files in the image_data directory
	# imageList = functGetFiles(path_to_image_data, 'jpg')
	# imageList = sorted(imageList)

	# get the list of annotation files in the image_data directory
	annotList = functGetFiles(path_to_image_data, 'txt')
	annotList = sorted(annotList)

	# imageListLen = len(imageList)
	# annotListLen = len(annotList)

	# print('imageListLen:', imageListLen)
	# print('annotListLen:', annotListLen)

	# print('imageList[0]:')
	# print(imageList[0])
	# print('imageList[imageListLen-1]:')
	# print(imageList[imageListLen-1])

	# print('annotList[0]:')
	# print(annotList[0])
	# print('annotList[annotListLen-1]:')
	# print(annotList[annotListLen-1])

	for annotPath in annotList:
		annot = read_annotation_file(annotPath)
		# print(annotPath)
		# print(annot)
		
		# read in the image:
		path2JpgIn = annotPath.replace('txt','jpg')
		filename = os.path.basename(path2JpgIn)

		# read as color
		img_rgb = cv2.imread(path2JpgIn,cv2.IMREAD_COLOR)

		for s in SCALE:
			# print('can_scale %d:' % (s), can_scale(annot,s))
			if can_scale(annot,s):
				# find ROI at scale s
				# roi = get_roi(annot,s)
				roi = get_roi2(annot,s)
				# print('s,roi:')
				# print(s,roi)

				# sys.exit()

				# scale the image
				image_scaled = astro_data.scale_image(img_rgb, s)

				# scale the ROI
				roi_scaled = scale_roi(roi,s)
				# print('roi_scaled:',roi_scaled)

				# extract ROI
				image_scaled_roi = astro_data.extract_roi(image_scaled,roi_scaled,IMAGE_WIDTH,IMAGE_HEIGHT)

				# save the scaled image
				filename_out = filename.replace('.jpg','-%d.jpg' % (s))
				path2JpgOut = os.path.join(path_to_image_data_s,filename_out)
				astro_data.save_image2(image_scaled_roi,path2JpgOut)

				# save the scaled and ROI-extracted annotation data
				# roi_annot_dict = scale_annot(annot,s,roi_scaled)
				roi_annot_dict = scale_annot2(annot,s,roi_scaled)
				annot_txt = scale_annot_2_txt(roi_annot_dict)
				if annot_txt != '':
					filename_annot_out = filename_out.replace('.jpg','.txt')
					path2AnnotOut = os.path.join(path_to_image_data_s,filename_annot_out)
					# print('path2AnnotOut:',path2AnnotOut)
					with open(path2AnnotOut, "w") as fptr:
						fptr.write(annot_txt)

				# save the scaled ground thruth image
				img_annot_rgb = astro_data.annotateImage2(image_scaled_roi,roi_annot_dict)
				path2JpgAnnot = os.path.join(path_to_image_data_gt,filename_out)
				astro_data.save_image2(img_annot_rgb,path2JpgAnnot)

				# sys.exit()

		# copy original image at scale 1X
		filename_out = filename.replace('.jpg','-1.jpg')
		path2JpgOut = os.path.join(path_to_image_data_s,filename_out)
		copyfile(path2JpgIn, path2JpgOut)

		# copy original annot at scale 1X
		filename_annot_out = filename_out.replace('.jpg','.txt')
		path2AnnotOut = os.path.join(path_to_image_data_s,filename_annot_out)
		copyfile(annotPath, path2AnnotOut)

		# create a ground truth image for scale 1X
		# img_rgb = cv2.imread(path2JpgIn,cv2.IMREAD_COLOR)
		img_annot_gt = astro_data.annotateImage2(img_rgb,annot)
		path2JpgGtOut = os.path.join(path_to_image_data_gt,filename_out)
		astro_data.save_image2(img_annot_gt,path2JpgGtOut)

		# sys.exit()

	print('Done')
	# sys.exit()

elif(args.stats == True):

	print("generate statistics on the average number of satellites detected per image frame")

	astro_data = Astro()
	
	# get the direrctories in which the FITS directories are contained:
	sourceDir =  os.path.abspath(jsonDir)
	print('sourceDir:', sourceDir)

	# load the parsed astro data:
	pickle_file = os.path.join(sourceDir, 'astroData.pickle')
	astro_data.loadAstroDataFromPickle(pickle_file)

	sourceDir = sourceDir.replace('/JSON','')
	print('fitsDir:', sourceDir)

	fitsDirs = sorted([f for f in os.listdir(sourceDir) if os.path.isdir(os.path.join(sourceDir, f))])
	print('fitsDir:', fitsDirs)

	fitsDirs = [ x for x in fitsDirs if "." in x ]
	print('fitsDirs:', fitsDirs)

	# for each directory, read the AST files in the directory of AST files converted to Json:
	print('Astrograph only:')
	dict_stats_dict = {}
	for d in fitsDirs:

		# create the output directory
		d_out = d.split('.')

		# if datan:
		d_dir_out = d_out[0] + '.' + d_out[2] + '.' + d_out[3] + '.' + d_out[1]
		# else:
		# 	d_dir_out = d_out[2] + '.' + d_out[3] + '.' + d_out[1]

		hist_dict = astro_data.getStats(d_dir_out)
		# print('directory:', d_dir_out, hist_dict, 'sum:', sum(hist_dict.items(), key=itemgetter(1)))
		# sum = 0
		# s = sum([v for k,v in hist_dict])
		print('directory:', d_dir_out, hist_dict,'sum:',sum([v for k,v in hist_dict]))		

		# accumulate the data:
		dict_stats_dict[d_dir_out] = hist_dict

	# correct histograms based on mechanical Turk input:
	print('Astrograph + Mechanical Turk:')

	turk_data = Turk()

	# get the direrctories in which the FITS directories are contained:
	sourceDir =  os.path.abspath(jsonDir)
	# print('sourceDir:', sourceDir)

	# load the parsed astro data:
	pickle_file = os.path.join(sourceDir, 'mechanicalTurk.pickle')
	turk_data.loadTurkDataFromPickle(pickle_file)

	getSourceDir = functCheckDir(jsonDir)
	#print('sourceDir:', getSourceDir)

	onlyDirs = sorted([f for f in os.listdir(getSourceDir) if os.path.isdir(os.path.join(getSourceDir, f))])

	# for each directory, read the "Mechanical Turk" file and the directory of AST files converted to Json:
	for d in onlyDirs:
		mechTurkFile = functGetFiles(os.path.join(getSourceDir, d),'json')
		mechTurkFile = mechTurkFile[0] # it's a list of files
		turk_file_name = mechTurkFile.split('/')[3]

		# print('turk_file_name:',turk_file_name)

		# find all the false negative detections from the Turk data:
		false_negatives_dict = turk_data.getFalseNegatives(turk_file_name)

		# find all the false positive detections from the Turk data:
		false_positives_dict = turk_data.getFalsePositives(turk_file_name)

		hist_dict = defaultdict(int,dict_stats_dict[d])

		# do for each false_negative entry:
		# count how many times each imagefilename appears in the list: this is the number of detections found in that image:
		num_false_negs_dict = Counter(false_negatives_dict['imagefilename'])
		for k,v in num_false_negs_dict.items():
			# get number of Astrograph detections:
			num_obs = astro_data.getNumObs(d,k)
			
			hist_dict[num_obs] -= 1

			hist_dict[num_obs+v] += 1

		# do for each false_positive entry:
		# remove the false positive detections from annotation data and update the Annotated Images:
		# count how many times each imagefilename appears in the list: this is the number of detections found in that image:
		num_false_pos_dict = Counter(false_positives_dict['imagefilename'])
		for k,v in num_false_pos_dict.items():
			# get number of Astrograph detections:
			num_obs = astro_data.getNumObs(d,k)
			
			hist_dict[num_obs] -= 1

			hist_dict[num_obs-v] += 1

		# remove all entries from the dictionary whose counts are zero:
		hist_dict_no_z = { k:v for k, v in hist_dict.items() if v }
		hist_dict = defaultdict(int,hist_dict_no_z)

		dict_stats_dict[d] = hist_dict

		# print('directory:', d, sorted(hist_dict.items(), key=itemgetter(0)), 'sum:', sum(hist_dict.items(), key=itemgetter(1)))
		print('directory:', d, sorted(hist_dict.items(), key=itemgetter(0)),'sum:',sum([v for k,v in hist_dict.items()]))

	nEntries = len(dict_stats_dict)
	print('Done:',nEntries)

	# Consistency check: count number of detections in each annotation file and generate a similar histogram
	print('Annotations:')
	sourceDir =  os.path.abspath(jsonDir)
	# print('sourceDir:', sourceDir)

	getSourceDir = functCheckDir(jsonDir)
	#print('sourceDir:', getSourceDir)

	onlyDirs = sorted([f for f in os.listdir(getSourceDir) if os.path.isdir(os.path.join(getSourceDir, f))])

	# print('dirs:', onlyDirs)

	# for each directory, read the "Mechanical Turk" file and the directory of AST files converted to Json:
	for f in onlyDirs:
		path2Annot = os.path.join(sourceDir, f, 'Annotations')
		# path2Images = os.path.join(sourceDir, f, 'ImageFilesAnnotated') # change to 'ImageFiles' after validating data visually
		hist_dict = defaultdict(int)
		# get the list of annotation files in the Annotations directory:
		annotationsList = sorted(functGetFiles(path2Annot,'txt'))
		for annot in annotationsList:
			# print('annot:',annot)
			# open the annotations file and read it:
			with open(annot) as fptr:
				file_str = fptr.read()
				# separate text into lines and parse each line
				text_list = file_str.split(str_new_line)
				# drop the last element in the list: '':
				# text_list = text_list[:-1]
				# print('text_list:',text_list)
				# print('len(text_list):',len(text_list))
				# skip the file if there are no annotations:
				num_obs =  len(text_list)
				# determine the number of occurrences of '' in the string:
				if '' in text_list:
					num_obs -=1
					# print('text_list:',text_list)
				# if num_obs == 1:
				# 	if text_list[0] == '':
				# 		num_obs = 0
				#count empty text as no observation:
				# num_obs -=1
				# if num_obs == 0:
				# 	print('annot:',annot)
				# 	print('file_str:', file_str)
				# 	print('text_list:', text_list)
					# sys.exit() 
				hist_dict[num_obs] += 1
		# print('directory:', f, sorted(hist_dict.items(), key=itemgetter(0)), 'sum:', sum(hist_dict.items(), key=itemgetter(1)))
		print('directory:', f, sorted(hist_dict.items(), key=itemgetter(0)),'sum:',sum([v for k,v in hist_dict.items()]))
		# sys.exit()

	print('Done:')
	# sys.exit()
elif(args.compare == True):

	print("compare two datasets for similar satellite detections")

	astro_data1 = Astro()  # uct-enabled
	astro_data2 = Astro()  # ground-truth
	
	# get the direrctories:
	jsonDir1 = args.dir
	jsonDir2 = args.directory2

	sourceDir1 =  os.path.abspath(jsonDir1)
	print('sourceDir1:', sourceDir1)

	sourceDir2 =  os.path.abspath(jsonDir2)
	print('sourceDir2:', sourceDir2)

	# load the parsed astro data:
	pickle_file1 = os.path.join(sourceDir1, 'astroData.pickle')
	astro_data1.loadAstroDataFromPickle(pickle_file1)

	pickle_file2 = os.path.join(sourceDir2, 'astroData.pickle')
	astro_data2.loadAstroDataFromPickle(pickle_file2)

	# get the list of directories in the data processed with the UCT flag enabled:
	jsonDirs1 = sorted([f for f in os.listdir(sourceDir1) if os.path.isdir(os.path.join(sourceDir1, f))])
	# print('jsonDirs1:')
	# for jDir1 in jsonDirs1:
	# 	print(jDir1)

	# jsonDirs1 = [ x for x in jsonDirs1 if "." in x ]
	# print('jsonDirs1:')
	# for jDir1 in jsonDirs1:
	# 	print(jDir1)

	# get the number of observations per each file in both the ground-truth and the UCT directories
	for jDir1 in jsonDirs1:
		obs_dict1 = astro_data1.getNumObsPerFile(jDir1)
		obs_dict2 = astro_data2.getNumObsPerFile(jDir1)
		print('Directory:',jDir1)
		print('filename:     n_gt  n_uct')

		# print(obs_dict1)
		for k in sorted(obs_dict1):
			print(k, obs_dict2[k], obs_dict1[k])
		print('')


	print('Done:')

else:
	print("Illegal command line option:")

