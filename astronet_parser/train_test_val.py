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
from random import shuffle

train_exclusion_list_start = ['abq01.12.09.2017_sat_41866.1200','abq01.12.10.2017_sat_41866.0007','abq01.12.11.2017_sat_41866.2633','rme02.04.22.2015_sat_36411.0001','rme02.04.22.2015_sat_36516.0010','rme02.07.01.2015_sat_26761.0045','rme02.07.01.2015_sat_26761.0271','rme02.07.01.2015_sat_26985.0037','rme02.07.01.2015_sat_28790.0061','rme02.07.02.2015_sat_36516.0019','rme02.07.02.2015_sat_37834.0002','rme02.07.03.2015_sat_26761.0001','rme02.07.03.2015_sat_26985.0001','rme02.07.03.2015_sat_28790.0001','rme02.07.04.2015_sat_36411.0037','rme02.07.04.2015_sat_36516.0013','rme02.07.04.2015_sat_37834.0187','rme02.07.07.2015_sat_26761.0002','rme02.07.07.2015_sat_26761.0325','rme02.07.07.2015_sat_26985.0001','rme02.07.07.2015_sat_28790.0001','rme02.07.14.2015_sat_36411.0164','rme02.07.14.2015_sat_36516.0007','rme02.07.14.2015_sat_37834.0007','rme02.07.15.2015_sat_26761.0025','rme02.07.15.2015_sat_26761.0259','rme02.07.15.2015_sat_26985.0019','rme02.07.15.2015_sat_28790.0031','rme02.07.16.2015_sat_36411.0013','rme02.07.16.2015_sat_36516.0105','rme02.07.17.2015_sat_26761.0007','rme02.07.17.2015_sat_26985.0008','rme02.07.17.2015_sat_28790.0001','rme02.07.22.2015_sat_36411.0007','rme02.07.22.2015_sat_36516.0019','rme02.07.23.2015_sat_26761.0001','rme02.07.23.2015_sat_26761.0223','rme02.07.23.2015_sat_26985.0079','rme02.07.23.2015_sat_26985.0163','rme02.07.23.2015_sat_28790.0007','rme02.07.24.2015_sat_36411.0001','rme02.07.24.2015_sat_36516.0055','rme02.07.25.2015_sat_26985.0055','rme02.07.25.2015_sat_28790.0007','rme02.07.28.2015_sat_36411.0001','rme02.07.28.2015_sat_36516.0001','rme02.07.29.2015_sat_36516.0283','rme02.07.30.2015_sat_36411.0001','rme02.07.30.2015_sat_36516.0037','rme02.07.31.2015_sat_26761.0002','rme02.07.31.2015_sat_26761.0211','rme02.07.31.2015_sat_26985.0267','rme02.07.31.2015_sat_28790.0001','rme02.08.01.2015_sat_36411.0025','rme02.08.01.2015_sat_36516.0201','rme02.08.07.2015_sat_36411.0043','rme02.08.07.2015_sat_36516.0067','rme02.08.08.2015_sat_26761.0007','rme02.08.08.2015_sat_26761.0134','rme02.08.08.2015_sat_26985.0124','rme02.08.08.2015_sat_28790.0019','rme02.08.11.2015_sat_36411.0013','rme02.08.11.2015_sat_36516.0019','rme02.08.11.2015_sat_37834.0229','rme02.08.12.2015_sat_26761.0013','rme02.08.12.2015_sat_26761.0199','rme02.08.12.2015_sat_26985.0019','rme02.08.12.2015_sat_28790.0043','rme02.08.14.2015_sat_26985.0079','rme02.08.14.2015_sat_28790.0062','rme02.08.21.2015_sat_36516.0019','rme02.08.26.2015_sat_26761.0001','rme02.08.26.2015_sat_26985.0001','rme02.08.26.2015_sat_28790.0001','rme02.08.28.2015_sat_26761.0001','rme02.08.28.2015_sat_26985.0031','rme02.08.28.2015_sat_28790.0007','rme02.08.29.2015_sat_36516.0158','rme02.10.23.2017_sat_36411.0714','rme02.10.30.2017_sat_36411.0001','rme02.10.30.2017_sat_36411.0727','rme03.10.26.2017_sat_36411.0019','rme03.10.27.2017_sat_36411.0487','rme03.10.28.2017_sat_36411.0487','rme03.10.31.2017_sat_36411.0473']
train_exclusion_list_stop  = ['abq01.12.09.2017_sat_41866.2545','abq01.12.10.2017_sat_41866.1135','abq01.12.11.2017_sat_41866.3174','rme02.04.22.2015_sat_36411.0168','rme02.04.22.2015_sat_36516.0213','rme02.07.01.2015_sat_26761.0114','rme02.07.01.2015_sat_26761.0288','rme02.07.01.2015_sat_26985.0318','rme02.07.01.2015_sat_28790.0324','rme02.07.02.2015_sat_36516.0234','rme02.07.02.2015_sat_37834.0006','rme02.07.03.2015_sat_26761.0102','rme02.07.03.2015_sat_26985.0150','rme02.07.03.2015_sat_28790.0323','rme02.07.04.2015_sat_36411.0162','rme02.07.04.2015_sat_36516.0274','rme02.07.04.2015_sat_37834.0240','rme02.07.07.2015_sat_26761.0108','rme02.07.07.2015_sat_26761.0330','rme02.07.07.2015_sat_26985.0330','rme02.07.07.2015_sat_28790.0336','rme02.07.14.2015_sat_36411.0179','rme02.07.14.2015_sat_36516.0264','rme02.07.14.2015_sat_37834.0135','rme02.07.15.2015_sat_26761.0054','rme02.07.15.2015_sat_26761.0330','rme02.07.15.2015_sat_26985.0312','rme02.07.15.2015_sat_28790.0330','rme02.07.16.2015_sat_36411.0017','rme02.07.16.2015_sat_36516.0204','rme02.07.17.2015_sat_26761.0066','rme02.07.17.2015_sat_26985.0060','rme02.07.17.2015_sat_28790.0138','rme02.07.22.2015_sat_36411.0012','rme02.07.22.2015_sat_36516.0294','rme02.07.23.2015_sat_26761.0084','rme02.07.23.2015_sat_26761.0336','rme02.07.23.2015_sat_26985.0090','rme02.07.23.2015_sat_26985.0336','rme02.07.23.2015_sat_28790.0330','rme02.07.24.2015_sat_36411.0054','rme02.07.24.2015_sat_36516.0276','rme02.07.25.2015_sat_26985.0336','rme02.07.25.2015_sat_28790.0342','rme02.07.28.2015_sat_36411.0083','rme02.07.28.2015_sat_36516.0276','rme02.07.29.2015_sat_36516.0318','rme02.07.30.2015_sat_36411.0066','rme02.07.30.2015_sat_36516.0282','rme02.07.31.2015_sat_26761.0041','rme02.07.31.2015_sat_26761.0342','rme02.07.31.2015_sat_26985.0335','rme02.07.31.2015_sat_28790.0330','rme02.08.01.2015_sat_36411.0066','rme02.08.01.2015_sat_36516.0256','rme02.08.07.2015_sat_36411.0066','rme02.08.07.2015_sat_36516.0275','rme02.08.08.2015_sat_26761.0011','rme02.08.08.2015_sat_26761.0275','rme02.08.08.2015_sat_26985.0264','rme02.08.08.2015_sat_28790.0300','rme02.08.11.2015_sat_36411.0066','rme02.08.11.2015_sat_36516.0298','rme02.08.11.2015_sat_37834.0300','rme02.08.12.2015_sat_26761.0036','rme02.08.12.2015_sat_26761.0300','rme02.08.12.2015_sat_26985.0306','rme02.08.12.2015_sat_28790.0306','rme02.08.14.2015_sat_26985.0215','rme02.08.14.2015_sat_28790.0294','rme02.08.21.2015_sat_36516.0288','rme02.08.26.2015_sat_26761.0012','rme02.08.26.2015_sat_26985.0078','rme02.08.26.2015_sat_28790.0024','rme02.08.28.2015_sat_26761.0036','rme02.08.28.2015_sat_26985.0360','rme02.08.28.2015_sat_28790.0366','rme02.08.29.2015_sat_36516.0301','rme02.10.23.2017_sat_36411.1147','rme02.10.30.2017_sat_36411.0006','rme02.10.30.2017_sat_36411.1362','rme03.10.26.2017_sat_36411.0335','rme03.10.27.2017_sat_36411.0834','rme03.10.28.2017_sat_36411.1458','rme03.10.31.2017_sat_36411.1487']


## CHECK INPUT DIRECTORY
def functCheckDir(inputDir):
    if not os.path.isdir(inputDir):
        print("\t\n ## Warning: Unable to find directory ##\n")
        sys.exit()
    else:
        return inputDir

## SELECT SPECIFIC FILE
def functGetFiles(inputDir, inExtension, fullPath = True):
    file_List = []
    for mainDir, subDir, fileName in os.walk(inputDir):
        for listFiles in os.listdir(mainDir):
            if listFiles.endswith(inExtension):
                if fullPath:
                    file_List.append(os.path.normpath(os.path.join(mainDir, listFiles)))
                else:
                    file_List.append(listFiles)
        return file_List
    
## WRITE TO FILE 
def functWriteToFile(inFile, inContent):
    fileWrite = open(inFile, "a")
    fileWrite.write(inContent + "\n")

    fileWrite.close()

def remove_train_exlusion_files(image_file_list):
	image_file_list_reduced = image_file_list[:]
	exclusion_list = []
	num_indices_total = 0

	#iterate through the start,stop train exclusion list:
	for str_start,str_stop in zip(train_exclusion_list_start, train_exclusion_list_stop):

		# print('str_start,str_stop',str_start, str_stop)

		image_file_list_reduced_reverse = image_file_list_reduced[:]
		image_file_list_reduced_reverse = list(reversed(image_file_list_reduced_reverse))

		# find the first index in the image_file_list that contains the str_start text:
		idx_begin = next(idx_strt for idx_strt,x in enumerate(image_file_list_reduced) if str_start in x) 
		idx_end   = next(idx_stop for idx_stop,x in enumerate(image_file_list_reduced_reverse) if str_stop in x)

		idx_end = len(image_file_list_reduced_reverse) - idx_end

		exclusion_list_to_append = image_file_list_reduced[idx_begin:idx_end]
		[exclusion_list.append(x) for x in exclusion_list_to_append]


		del image_file_list_reduced[idx_begin:idx_end]

		num_indices = idx_end - idx_begin
		num_indices_total += num_indices

		# print('image_file_list[idx_begin:idx_end]',image_file_list[idx_begin:idx_end])

		# print('num_indices:',num_indices)
		# print('len(image_file_list_reduced), len(exclusion_list)',len(image_file_list_reduced), len(exclusion_list))

		print(str_start, str_stop, len(image_file_list), len(image_file_list_reduced), len(exclusion_list), num_indices, num_indices_total)

	return image_file_list_reduced, exclusion_list

## MAIN FUNCTION
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-g","--generate", action="store_true", help="generate training, validation and test text files")
	parser.add_argument("-c","--correct",  action="store_true", help="correct validation and test text files by removing exclusion_list data and keeps as a separate list")
	parser.add_argument("-d","--dir", type=str, default='../datat/image_data_s', help="path to image_data dir")
	parser.add_argument("-o","--output", type=str, default='../datat', help="path to output dir")
	args = parser.parse_args()

	train_frac = 0.80
	validate_frac = 0.10
	test_frac = 0.10

if(args.generate == True):
	print('generate training data:...')

	# where to store the output (train,tesst,valid).txt files:
	outputDir = os.path.abspath(args.output)
	print('output_path:',outputDir)

	image_data_path = os.path.abspath(args.dir)
	print('image_data_path:',image_data_path)

	image_file_list = sorted(functGetFiles(image_data_path,'jpg',False))
	nFiles = len(image_file_list)
	print('nFiles', nFiles)

	# remove those files that can't be used for training: the train exclusion list:
	image_file_list_reduced, exclusion_list = remove_train_exlusion_files(image_file_list)

	# shuffle the exclusion list:
	# shuffle(image_file_list_reduced)
	shuffle(exclusion_list)

	nFiles = len(image_file_list_reduced)
	nFilesExcluded = len(exclusion_list)
	print('nFiles:image_file_list_reduced', nFiles)
	print('nFilesExcluded:exclusion_list', nFilesExcluded)

	# sys.exit()

	# determine the number of files for training, validation and test:
	nTrain = round(train_frac * nFiles)
	nValid = round(validate_frac * nFiles)
	nTest = round(test_frac * nFiles)

	# print('nFiles, nTrain, nValid, nTest')
	# print(nFiles, nTrain, nValid, nTest)

	if(nFiles != (nTrain + nValid + nTest)):
		nSum = nTrain + nValid +nTest
		nTest = nFiles - nSum + nTest

	nValidAndTest = nValid+nTest
	print('nFiles, nTrain, nValid, nTest, nValidAndTest:')
	print(nFiles, nTrain, nValid, nTest, nValidAndTest)

	# generate random indices. for training:
	# idx_files = np.arange(nFiles)
	idx_train = np.random.choice(nFiles, nTrain, False)
	# print('len(idx_train):',len(idx_train))

	idx_train_list = idx_train.tolist()
	# print('idx_train_list:')
	# print(idx_train_list)

	# image_file_train_list = image_file_list[idx_train_list]
	image_file_train_list = [image_file_list_reduced[index] for index in idx_train_list]
	# print('len(image_file_train_list):',len(image_file_list_reduced))
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
	idx_valid_test_list = [x for x in idx_list if x not in idx_train_list]
	print('len(idx_valid_test_list)', len(idx_valid_test_list))

	# randomly shuffle the list of indices:
	shuffle(idx_valid_test_list)

	# allocate the first nValid indices to valid.txt and the indices (nTest) to test.txt
	idx_split = nValid
	idx_valid_list  = idx_valid_test_list[:idx_split]
	idx_test_list  = idx_valid_test_list[idx_split:]

	print('len(idx_valid_list:',len(idx_valid_list))
	print('len(idx_test_list:',len(idx_test_list))

	# save vaild set to file:
	image_file_valid_list = [image_file_list_reduced[index] for index in idx_valid_list]
	# print('len(image_file_valid_list):',len(image_file_valid_list))

	path2Valid = os.path.join(outputDir,'valid.txt')
	print(path2Valid)

	# Equally divide the exclusion list among validation and test data
	nExclusionList = len(exclusion_list)
	nValidE = round(0.5 * nExclusionList)
	nTestE = round(0.5* nExclusionList)

	# print('nExclusionList, nValidE, nTestE')
	# print(nExclusionList, nValidE, nTestE)

	if(nExclusionList != (nValidE + nTestE)):
		nSumE = nValidE +nTestE
		nTestE = nExclusionList - nSumE + nTestE

	print('nExclusionList, nValidE, nTestE')
	print(nExclusionList, nValidE, nTestE)

	# generate random indices for validation from the training_exclusion_list:
	idx_valid_e = np.random.choice(nExclusionList, nValidE, False)
	# print('len(idx_valid_e):',len(idx_valid_e))

	idx_valid_e_list = idx_valid_e.tolist()
	# print(idx_valid_e_list)

	image_file_valid_e_list = [exclusion_list[index] for index in idx_valid_e_list]
	# print('len(image_file_valid_e_list):',len(image_file_valid_e_list))

	# concatenate the two lists:
	image_file_valid_list_concat = image_file_valid_list + image_file_valid_e_list
	# print('len(image_file_valid_list_concat):',len(image_file_valid_list_concat))

	with open(path2Valid, "w") as fptr:
		for f in image_file_valid_list_concat:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)

	# get the remaining list of indices, after having removed the validation indices from the exclusion list of indices:
	idx_list_e = np.arange(nExclusionList)
	idx_test_e_list = [x for x in idx_list_e if x not in idx_valid_e_list]

	# randomly shuffle the list of indices:
	shuffle(idx_test_e_list)

	# save test set to file:
	image_file_test_list = [image_file_list_reduced[index] for index in idx_test_list]
	print('len(image_file_test_list):',len(image_file_test_list))

	image_file_test_e_list = [exclusion_list[index] for index in idx_test_e_list]
	print('len(image_file_test_e_list):',len(image_file_test_e_list))

	# concatenate the two lists:
	image_file_test_list_concat = image_file_test_list + image_file_test_e_list
	print('len(image_file_test_list_concat):',len(image_file_test_list_concat))

	path2Test = os.path.join(outputDir,'test.txt')
	print(path2Test)

	with open(path2Test, "w") as fptr:
		for f in image_file_test_list_concat:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)

	print('nTrain:', nTrain)
	print('nValid:', len(image_file_valid_list_concat),'=', nValid, '+',len(image_file_valid_e_list))
	print(' nTest:', len(image_file_test_list_concat), '=', nTest,  '+',len(image_file_test_e_list))
	print('nTotal:', nTrain + len(image_file_valid_list_concat) + len(image_file_test_list_concat))

	print('Done')

elif(args.correct == True):
	print('correct validate and test data by removing the exclusion_list:...')

	# where to store the output (train,tesst,valid).txt files:
	outputDir = os.path.abspath(args.output)
	print('output_path:',outputDir)

	image_data_path = os.path.abspath(args.dir)
	print('image_data_path:',image_data_path)

	image_file_list = sorted(functGetFiles(image_data_path,'jpg',False))
	nFiles = len(image_file_list)
	print('nFiles', nFiles)

	# remove those files that can't be used for training: the train exclusion list:
	image_file_list_reduced, exclusion_list = remove_train_exlusion_files(image_file_list)

	path2Valid = os.path.join(outputDir,'valid.txt')
	print(path2Valid)

	# Equally divide the exclusion list among validation and test data
	nExclusionList = len(exclusion_list)

	# read in the valid path list:
	with open(path2Valid) as fptr:
		valid_str = fptr.read()

	valid_path_list = valid_str.split('\n')
	# remove the last '' element from the list
	valid_path_list = valid_path_list[:-1]

	nValid = len(valid_path_list)

	# remove all exclusion_list items from the valid_path_list:
	for  excl_item in exclusion_list:
		if excl_item in valid_path_list:
			valid_path_list.remove(excl_item)

	nValidE = len(valid_path_list)

	path2Valid_exclude = os.path.join(outputDir,'valid_exclude.txt')
	print(path2Valid_exclude)

	with open(path2Valid_exclude, "w") as fptr:
		for f in valid_path_list:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)


	path2Test = os.path.join(outputDir,'test.txt')
	print(path2Test)

	# read in the test path list:
	with open(path2Test) as fptr:
		test_str = fptr.read()

	test_path_list = test_str.split('\n')
	# remove the last '' element from the list
	test_path_list = test_path_list[:-1] 

	nTest = len(test_path_list)

	# remove all exclusion_list items from the test_path_list:
	for  excl_item in exclusion_list:
		if excl_item in test_path_list:
			test_path_list.remove(excl_item)

	nTestE = len(test_path_list)

	path2Test_exclude = os.path.join(outputDir,'test_exclude.txt')
	print(path2Test_exclude)

	with open(path2Test_exclude, "w") as fptr:
		for f in test_path_list:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)

	path2Exclude = os.path.join(outputDir,'exclude.txt')
	print(path2Exclude)

	with open(path2Exclude, "w") as fptr:
		for f in exclusion_list:
			str_out_valid = f + '\n'
			fptr.write(str_out_valid)

	print('nExclusionList:', nExclusionList)
	print('nValid:', nValid)
	print('nValidE:', nValidE)
	print('nTest:', nTest)
	print('nTestE:', nTestE)

	print('Done')


else:
	print("Illegal command line option:")

