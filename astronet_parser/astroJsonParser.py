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
from jsonParserTurk import Turk
from jsonParserAstro import Astro

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

## MAIN FUNCTION
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d","--dir", type=str, default='../data/JSON', help="path to JSON data dir")
	parser.add_argument("-p","--parse", type=str, help="parse Astro or Turk data")
	parser.add_argument("-a","--annot", action="store_true", help="generate annotation data")
	parser.add_argument("-t","--turk2annot", action="store_true", help="update annotation data with human turk data")
	parser.add_argument("-k","--darknetTrain", action="store_true", help="generate data for training via darknet YOLO.v2 model")
	# totalArg = len(sys.argv)
	# if(totalArg != 3):
	# 	print("\n +++  Verify Number of Arguments +++ \n")
	# 	print("Command Format: python astroJsonParser.py <Json Folder > <parseTurk or parseAstro\n")
	# 	sys.exit()
	args = parser.parse_args()
	#jsonDir = str(sys.argv[1])
	jsonDir = args.dir
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

	str_csv_separator = ' '
	str_new_line = '\n'

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
		mechTurkFile = functGetFiles(os.path.join(getSourceDir, f),'json')
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
		d_dir_out = d_out[2] + '.' + d_out[3] + '.' + d_out[1]

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
		# print(nFiles)
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

else:
	print("Illegal command line option:")

