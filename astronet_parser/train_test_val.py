## train_test_val.py
## gmartin @ 02/22/2018
## INPUT:
##      Generate data text files for training, validation and test 
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
    
## WRITE TO FILE 
def functWriteToFile(inFile, inContent):
    fileWrite = open(inFile, "a")
    fileWrite.write(inContent + "\n")

    fileWrite.close()

## MAIN FUNCTION
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-g","--generate", action="store_true", help="generate training, validation and test text files")
	parser.add_argument("-d","--dir", type=str, default='../data/image_data', help="path to image_data dir")
	parser.add_argument("-o","--output", type=str, default='../data', help="path to output dir")
	args = parser.parse_args()

	train_frac = 0.80
	validate_frac = 0.10
	test_frac = 0.10

	image_data_dir_name = 'image_data'

if(args.generate == True):
	print('generate training data:...')

	# where to store the output (train,tesst,valid).txt files:
	outputDir = os.path.abspath(args.output)
	print('output_path:',outputDir)

	image_data_path = os.path.abspath(args.dir)
	print('image_data_path:',image_data_path)

	image_file_list = functGetFiles(image_data_path,'jpg')
	nFiles = len(image_file_list)
	print('nFiles', nFiles)

	# determine the number of files for training, validation and test:
	nTrain = round(train_frac * nFiles)
	nValid = round(validate_frac * nFiles)
	nTest = round(test_frac * nFiles)

	print('nFiles, nTrain, nValid, nTest')
	print(nFiles, nTrain, nValid, nTest)

	if(nFiles != (nTrain + nValid + nTest)):
		nSum = nTrain + nValid +nTest
		nTest = nFiles - nSum + nTest


	print('nFiles, nTrain, nValid, nTest')
	print(nFiles, nTrain, nValid, nTest)

	# generate random indices. for training:
	# idx_files = np.arange(nFiles)
	idx_train = np.random.choice(nFiles, nTrain, False)
	print('len(idx_train):',len(idx_train))

	idx_train_list = idx_train.tolist()
	# print(idx_train_list)

	# image_file_train_list = image_file_list[idx_train_list]
	image_file_train_list = [image_file_list[index] for index in idx_train_list]
	print('len(image_file_train_list):',len(image_file_train_list))
	# print(image_file_train_list[0])
	# print(image_file_train_list[1])

	# save training data list to train.txt
	path2Train = os.path.join(outputDir,'train.txt')
	print(path2Train)

	with open(path2Train, "w") as fptr:
		for f in image_file_train_list:
			str_out_train = f + '\n'
			fptr.write(str_out_train)

	# get the remaining list of indices, after having removed the training indices:
	idx_list = np.arange(nFiles)
	idx_valid_test_list = [x for x in idx_list if x not in idx_train]
	print(len(idx_valid_test_list))

	# allocate the first nValid indices to valid.txt and the indices (nTest) to test.txt
	idx_split = nValid
	idx_valid_list  = idx_valid_test_list[:idx_split]
	idx_test_list  = idx_valid_test_list[idx_split:]

	print('len(idx_valid_list:',len(idx_valid_list))
	print('len(idx_test_list:',len(idx_test_list))

	# save vaild set to file:
	image_file_valid_list = [image_file_list[index] for index in idx_valid_list]
	print('len(image_file_valid_list):',len(image_file_valid_list))

	path2Valid = os.path.join(outputDir,'valid.txt')
	print(path2Valid)

	with open(path2Valid, "w") as fptr:
		for f in image_file_valid_list:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)

	# save test set to file:
	image_file_test_list = [image_file_list[index] for index in idx_test_list]
	print('len(image_file_test_list):',len(image_file_test_list))

	path2Test = os.path.join(outputDir,'test.txt')
	print(path2Test)

	with open(path2Test, "w") as fptr:
		for f in image_file_test_list:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)

	print('Done')


else:
	print("Illegal command line option:")

