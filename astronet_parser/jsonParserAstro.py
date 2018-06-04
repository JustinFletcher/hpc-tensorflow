__author__ = 'gmartin'
__version__ = '1.0'
# Interface for accessing the Astronomical dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import cv2
import astropy.io.fits
import json
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy
import itertools
#from . import mask as maskUtils
import os
from collections import defaultdict
from operator import itemgetter
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve
import pickle


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Astro:
    def __init__(self, annotation_file=None):
        """
        Constructor of Astronomical helper class for reading and visualizing astronomical annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.astroDict = dict()
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
#         if not annotation_file == None:
#             print('loading Turk annotations into memory...')
#             print(annotation_file)
#             tic = time.time()
#             dataset = json.load(open(annotation_file, 'r'))
#             assert type(dataset)==dict, 'Turk annotation file format {} not supported'.format(type(dataset))
#             print('Done (t={:0.2f}s)'.format(time.time()- tic))
#             self.dataset = dataset
# #           self.createIndex()
#             print('dataset:')
#             print(len(dataset)) # it's a dictionary of length 1
#            # get the list of individual 'paragraphs':
#            #paragraph_list = dataset[0]
#             for key in dataset.keys():
#                 print(len(dataset[key]))
#                 dataset_list = dataset[key]
#                #print(isinstance(dataset_list, list))
#                 for index,elem in enumerate(dataset_list):
#                     print()
#                     print(index)
#                     for k,v in elem.items():
#                         print(k,v)


    def addAstroFile(self, astro_file_list):
        if astro_file_list:
            astro_file_name = astro_file_list[0].split('/')[3]
            # print('astro_file_list:')
            # print(astro_file_list)
            # print('astro_file_name:',astro_file_name)

            self.astroDict[astro_file_name] = dict()

            # add the AST keys to the dictionary:
            self.astroDict[astro_file_name]['catalog_star_astrometry'] = {"value": []}
            self.astroDict[astro_file_name]['catalog_star_match_results'] = {   "center_dec_bias": [],
                                                                                "center_ra_bias": [],
                                                                                "projection": [],
                                                                                "rotation_angle": [],
                                                                                "true_center_dec": [],
                                                                                "true_center_ra": [] }
            self.astroDict[astro_file_name]['detected_star_astrometry'] = {"value": []}
            self.astroDict[astro_file_name]['differential_photometry'] = {  "photometric_stars": [],
                                                                            "zeropoint": [],
                                                                            "zeropoint_stdev": []}
            self.astroDict[astro_file_name]['header'] = {   "astroversion": [],
                                                            "b3_files": [],
                                                            "catalog_stars_in_fov": [],
                                                            "catalog_stars_matched": [],
                                                            "created_by": [],
                                                            "created_on": [],
                                                            "detected_stars_in_fov": [],
                                                            "highest_coeff_fit": [],
                                                            "image_airmass": [],
                                                            "image_exposure": [],
                                                            "image_filter": [],
                                                            "image_group": [],
                                                            "image_header_dec": [],
                                                            "image_header_ra": [],
                                                            "image_start_date": [],
                                                            "image_start_time": [],
                                                            "input_file": [],
                                                            "object_class": [],
                                                            "object_light_time_correction": [],
                                                            "object_name": [],
                                                            "object_velocity_aberration": [],
                                                            "objects": [],
                                                            "obs_files": [],
                                                            "photometry_method": [],
                                                            "reference_frame": [],
                                                            "sensor_fov": [],
                                                            "sensor_id": [],
                                                            "sensor_name": [],
                                                            "sensor_size": [],
                                                            "site": [],
                                                            "star_catalog": [],
                                                            "star_velocity_aberration": [],
                                                            "stddev": [],
                                                            "target_detect_method": [],
                                                            "track_mode": [],
                                                            "value": [],
                                                            'eossa_files': [],
                                                            'preraft_files': [],
                                                            'image_center_azel': [],
                                                            'azel_bounds_degs': [],
                                                            'azel_origin_degs': [],
                                                            'calibrations': [],
                                                            'frame_processing_status': [],
                                                            'height_m': [],
                                                            'latitude_deg': [],
                                                            'longitude_deg': [],
                                                            'mean_star_spot_size': [],
                                                            'pixel_bounds': [],
                                                            'pixel_origin': [],
                                                            'processing_status': [],
                                                            'radec_bounds_degs': [],
                                                            'sensor_temperature': [],
                                                            'star_processing': [],
                                                            'std_star_spot_size': [],
                                                            'task_description': [],
                                                            'gtds_files': [],
                                                            'nominal_pointing_offsets': [],
                                                            'predictions': [],
                                                            'collection_uuid': []}
            # self.astroDict[astro_file_name]['object_astrometry'] = {    "classification": [],
            #                                                             "date": [],
            #                                                             "filter": [],
            #                                                             "fwhm": [],
            #                                                             "image_dec": [],
            #                                                             "image_flux": [],
            #                                                             "image_instmag": [],
            #                                                             "image_mag": [],
            #                                                             "image_magerr": [],
            #                                                             "image_ra": [],
            #                                                             "name": [],
            #                                                             "refframe": [],
            #                                                             "tasking": [],
            #                                                             "time": [],
            #                                                             "x_center": [],
            #                                                             "y_center": [],
            #                                                             "value": []}
            self.astroDict[astro_file_name]['detections'] = {'object_ast': []}
            self.astroDict[astro_file_name]['three_coefficient_plate_solution'] = {
                                                                                    "fit_error_chi2x": [],
                                                                                    "fit_error_chi2y": [],
                                                                                    "fit_error_nstars": [],
                                                                                    "fit_error_total": [],
                                                                                    "x_cd11": [],
                                                                                    "x_cd11_error": [],
                                                                                    "x_cd12_error": [],
                                                                                    "x_cd21": [],
                                                                                    "xbias_error": [],
                                                                                    "xbias_xi": [],
                                                                                    "y_cd21": [],
                                                                                    "y_cd21_error": [],
                                                                                    "y_cd22": [],
                                                                                    "y_cd22_error": [],
                                                                                    "ybias_error": [],
                                                                                    "ybias_eta": []}

            self.object_astrometry = {  "classification": None,
                                        "date": None,
                                        "filter": None,
                                        "fwhm": None,
                                        "image_dec": None,
                                        "image_flux": None,
                                        "image_instmag": None,
                                        "image_mag": None,
                                        "image_magerr": None,
                                        "image_ra": None,
                                        "name": None,
                                        "refframe": None,
                                        "tasking": None,
                                        "time": None,
                                        "x_center": None,
                                        "y_center": None,
                                        "value": None }


            for f in astro_file_list:
                # print('f:',f)
                dataset = json.load(open(f, 'r'))
                assert type(dataset)==dict, 'Astro annotation file format {} not supported'.format(type(dataset))
                self.dataset = dataset

                object_ast_list = []
                for k,v in dataset.items():
                    if k == 'object_astrometry':
                        # check if the value is a list or simply a number of dictionary elements
                        if isinstance(v, list):
                            # print('in list:',v)
                            # print('len(v):',len(v))
                            for obj_ast_list_item in v:
                                # obj_ast_list_item is a of dictionary items: need to access each dictionary:
                                # print('obj_ast_list_item:')
                                # print(obj_ast_list_item)
                                if 'value' in obj_ast_list_item.keys():
                                    # print('obj_ast_dict.keys():')
                                    # print(obj_ast_list_item.keys())
                                    for k1,v1 in self.object_astrometry.items():
                                        # print('k1,v1:',k1,v1)
                                        self.object_astrometry[k1] = False
                                    self.object_astrometry['value'] = True
                                    object_ast_list.append(copy.deepcopy(self.object_astrometry))
                                else:       
                                    # print('obj_ast_list_item.keys():')
                                    # print(obj_ast_list_item.keys())
                                    for k1,v1 in obj_ast_list_item.items():
                                        # print('k1,v1:',k1,v1)
                                        if k1 in self.object_astrometry.keys():
                                            self.object_astrometry[k1] = v1
                                        else:
                                            print('{} not in object_astronomy dictionary'.format(k1))
                                    self.object_astrometry['value'] = False
                                    # print('self.object_astrometry:')
                                    # print(self.object_astrometry)
                                    object_ast_list.append(copy.deepcopy(self.object_astrometry))
                                    # print('object_ast_list:')
                                    # print(object_ast_list)
                                    # print('object_ast_list len:',len(object_ast_list))
                        else:
                            # print('in not a list')
                            if 'value' in v.keys():
                                # print('v.keys():')
                                # print(v.keys())
                                for k1,v1 in self.object_astrometry.items():
                                    # print('k1,v1 in self.object_astronomy:',k1,v1)
                                    self.object_astrometry[k1] = False
                                self.object_astrometry['value'] = True
                            else:       
                                for k1,v1 in v.items():
                                    # print('k1,v1 in v.items():',k1,v1)
                                    if k1 in self.object_astrometry:
                                        self.object_astrometry[k1] = v1
                                    else:
                                        print('{} not in object_astronomy dictionary'.format(k1))
                                self.object_astrometry['value'] = False
                            object_ast_list.append(copy.deepcopy(self.object_astrometry))
                        # self.astroDict[astro_file_name]['detections']['object_ast'].append(object_ast_list)

                    # if "value" is a key in this key, then set all values for all keys 'False'
                    elif 'value' in v.keys() and k != 'header' and k != 'object_astrometry':
                        for k1,v1, in self.astroDict[astro_file_name][k].items():
                            #print('{}: {}'.format(k1, v1))
                            if k1 == 'value':
                                self.astroDict[astro_file_name][k][k1].append(True)
                            else:
                                self.astroDict[astro_file_name][k][k1].append(False)
                    else:
                        for k1,v1, in v.items():
                            #print('{}: {}'.format(k1, v1))
                            if k1 in self.astroDict[astro_file_name][k]:
                                self.astroDict[astro_file_name][k][k1].append(v1)
                            else:
                                print('{} not in astroDict[{}]'.format(k1,k))
                        # some fields in 'header' maybe missing. Check and append with false if true
                        if k == 'header':
                            if 'b3_files' not in v.keys():
                                self.astroDict[astro_file_name][k]['b3_files'].append(False)
                            if 'obs_files' not in v.keys():
                                self.astroDict[astro_file_name][k]['obs_files'].append(False)
                            if 'target_detect_method' not in v.keys():
                                self.astroDict[astro_file_name][k]['target_detect_method'].append(False)
                            if 'eossa_files' not in v.keys():
                                self.astroDict[astro_file_name][k]['eossa_files'].append(False)
                            if 'preraft_files' not in v.keys():
                                self.astroDict[astro_file_name][k]['preraft_files'].append(False)
                            if 'image_center_azel' not in v.keys():
                                self.astroDict[astro_file_name][k]['image_center_azel'].append(False)
                            if 'azel_bounds_degs' not in v.keys():
                                self.astroDict[astro_file_name][k]['azel_bounds_degs'].append(False)
                            if 'azel_origin_degs' not in v.keys():
                                self.astroDict[astro_file_name][k]['azel_origin_degs'].append(False)
                            if 'calibrations' not in v.keys():
                                self.astroDict[astro_file_name][k]['calibrations'].append(False)
                            if 'frame_processing_status' not in v.keys():
                                self.astroDict[astro_file_name][k]['frame_processing_status'].append(False)
                            if 'height_m' not in v.keys():
                                self.astroDict[astro_file_name][k]['height_m'].append(False)
                            if 'latitude_deg' not in v.keys():
                                self.astroDict[astro_file_name][k]['latitude_deg'].append(False)
                            if 'longitude_deg' not in v.keys():
                                self.astroDict[astro_file_name][k]['longitude_deg'].append(False)
                            if 'mean_star_spot_size' not in v.keys():
                                self.astroDict[astro_file_name][k]['mean_star_spot_size'].append(False)
                            if 'pixel_bounds' not in v.keys():
                                self.astroDict[astro_file_name][k]['pixel_bounds'].append(False)
                            if 'pixel_origin' not in v.keys():
                                self.astroDict[astro_file_name][k]['pixel_origin'].append(False)
                            if 'processing_status' not in v.keys():
                                self.astroDict[astro_file_name][k]['processing_status'].append(False)
                            if 'radec_bounds_degs' not in v.keys():
                                self.astroDict[astro_file_name][k]['radec_bounds_degs'].append(False)
                            if 'sensor_temperature' not in v.keys():
                                self.astroDict[astro_file_name][k]['sensor_temperature'].append(False)
                            if 'star_processing' not in v.keys():
                                self.astroDict[astro_file_name][k]['star_processing'].append(False)
                            if 'std_star_spot_size' not in v.keys():
                                self.astroDict[astro_file_name][k]['std_star_spot_size'].append(False)
                            if 'task_description' not in v.keys():
                                self.astroDict[astro_file_name][k]['task_description'].append(False)
                            if 'gtds_files' not in v.keys():
                                self.astroDict[astro_file_name][k]['gtds_files'].append(False)
                            if 'nominal_pointing_offsets' not in v.keys():
                                self.astroDict[astro_file_name][k]['nominal_pointing_offsets'].append(False)
                            if 'predictions' not in v.keys():
                                self.astroDict[astro_file_name][k]['predictions'].append(False)
                            if 'collection_uuid' not in v.keys():
                                self.astroDict[astro_file_name][k]['collection_uuid'].append(False)

                self.astroDict[astro_file_name]['detections']['object_ast'].append(object_ast_list)

            print()
            self.info(astro_file_name,True)
            return

    def saveAstroDataAsPickle(self, pickle_file):
        #pass
        # Store data (serialize)
        print('Saving pickle file...')
        with open(pickle_file, 'wb') as handle:
            pickle.dump(self.astroDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        # verify save:
        # print(self.astroDict == self.loadAstroDataFromPickle(pickle_file))

    def loadAstroDataFromPickle(self, pickle_file):
        #pass
        # Load data (deserialize)
        print('Loading pickle file...')
        with open(pickle_file, 'rb') as handle:
            unserialized_data = pickle.load(handle)
            handle.close()
            # return unserialized_data
            self.astroDict = unserialized_data
       
    def info(self,whichDictionary='All',printLen=True):
        """
        Print information about the astro annotation file.
        :return:
        """
        if whichDictionary == 'All':
            totalCount = 0;
            for k in self.astroDict.keys():
                print()
                print(k,':')
                for k1,v1 in self.astroDict[k].items():
                    print(k1,':')
                    for k2,v2 in self.astroDict[k][k1].items():
                        nElem = len(v2)
                        if printLen:
                            print('{}: {}'.format(k2, len(v2)))
                        else:
                            print('{}: {}'.format(k2, v2))
                    print()
                totalCount += nElem
            print('Total entries: {}'.format(totalCount))
            print('Total collections: {}'.format(totalCount/6.))
        else:
            print()
            print(whichDictionary,':')
            for k1,v1 in self.astroDict[whichDictionary].items():
                print(k1,':')
                for k2,v2 in self.astroDict[whichDictionary][k1].items():
                    if printLen:
                        print('{}: {}'.format(k2, len(v2)))
                    else:
                        print('{}: {}'.format(k2, v2))
                print()

    def read_fits(self,filepath):
        """Reads simple 1-hdu FITS file into a numpy arrays
        
        A transposition is done, so that the indexes [x,y] of the numpy array follow the orientation of x and y in DS9
        and SExtractor.
        
        Parameters
        ----------
        filepath : str
            Filepath to read the array from
        
        """
        # astropy.io.fits.info(filepath)

        # a = astropy.io.fits.getdata(filepath).transpose()
        a = astropy.io.fits.getdata(filepath)
        # logger.info("Read FITS images %s from file %s" % (a.shape, filepath))
        # print('a.shape:', a.shape)

        a = a.astype(np.uint16)
        # fig = plt.figure()
        # num_bins = 256
        # n, bins, patches = plt.hist(a, num_bins, facecolor='blue', alpha=0.5)
        # plt.title("Histogram")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
        # # plt.show()
        # plt.savefig('histogram.png')

        # plt.hist(a.flatten(), bins=65536, range=(0.0, 1.0), fc='k', ec='k')
        # # plt.show()

        # fig = plt.gcf()

        # plot_url = py.plot_mpl(fig, filename='basic-histogram')

        # a = a.byteswap()
        # # a = cv2.equalizeHist(a)
        # # a = cv2.normalize(src = a, dst = None, alpha = 256, beta = 65535, norm_type = cv2.NORM_MINMAX);
        # a = a.byteswap()
        # # np.right_shift(a, 8)
        # # a= cv2.bitwise_not(a)
        a = self.histogram_equalize(a,False)
        # np.left_shift(a, 8)
        # print('np.max(a):',np.max(a))
        # print('np.min(a):',np.min(a))
        a = (a/256).astype(np.uint8)
        # print('np.max(a):',np.max(a))
        # print('np.min(a):',np.min(a))
        # a = cv2.equalizeHist(a)
        # a = histogram_equalize(a)
        # a = a.byteswap()

        # print('np.shape:', np.shape(a))
        # print('a size:', a.size)
        # print('a type:', a.dtype)
        # print('a bytes:', a.nbytes)

        # np.right_shift(a, 8)
        # np.left_shift(a, 8)

        # a= cv2.bitwise_not(a)

        # a = a.astype(np.uint8)

        # a= cv2.bitwise_not(a)

        # print('a size:', a.size)
        # print('a type:', a.dtype)
        # print('a bytes:', a.nbytes)


        return a

    def save_image(self, data, fn):

        cv2.imwrite(fn, data)

        # numpngw.write_png(fn, data)

    def save_image2(self, data, fn):

        sizes = np.shape(data)
        # print('fn:',fn)
        # print('sizes:',sizes)
        height = float(sizes[0])
        width = float(sizes[1])
         
        fig = plt.figure()
        fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(data,cmap='gray')
        plt.savefig(fn, dpi = height) 
        plt.close()

    def histogram_equalize(self, data, eightBits = True):
        #img = cv2.imread('tsukuba_l.png',0)
        sizes = np.shape(data)
        height = sizes[0]
        width = sizes[1]

        # print('sizes:',sizes)

        # print("h,w: {} and {}".format(height,width))

        img = np.zeros([height,width,3])

        img = data

        # max_L = 2**16-1
        # print("max_L: {}".format(max_L))

        # img[:,:,0] = np.ones([height,width])*max_L
        # img[:,:,1] = np.ones([height,width])*max_L
        # img[:,:,2] = np.ones([height,width])*max_L

        # print('img size:', img.size)
        # print('img type:', img.dtype)
        # print('img bytes:', img.nbytes)

        # img = img/256.
        if eightBits == True:
            depth = 256
        else:
            depth = 65536

        # np.right_shift(img, 8)
        # np.left_shift(a, 8)

        # img = cv2.bitwise_not(img)

        # img = img.astype(np.uint16)
        # img = img.astype(np.uint8)

        # print('img size:', img.size)
        # print('img type:', img.dtype)
        # print('img bytes:', img.nbytes)

        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img

        # create a CLAHE object (Arguments are optional).
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1,1))
        # img_gray = clahe.apply(img_gray)

        # img_gray = cv2.equalizeHist(img_gray)
        # img_gray = cv2.normalize(src = img_gray, dst = None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX);

        # invert the gray scale:
        #img_equal_gray = cv2.bitwise_not(img_equal_gray) 

        if eightBits == True:
            lut = self.histogram_clip(img_gray,0.1,0.999,depth,31,248,True)
        else:
            lut = self.histogram_clip(img_gray,0.2, 0.99,depth,31*256,248*256,False)
        # print("lut size:",lut.size)

        # img_equal_gray = cv2.LUT(img_gray, lut)
        img_equal_gray = self.apply_lut(img_gray,lut)
        # img_equal_gray = img_gray

        img_equal_rgb = cv2.cvtColor(img_equal_gray, cv2.COLOR_GRAY2RGB)

        return img_equal_rgb
        # return img_equal_gray

        #cv2.imwrite('clahe_2.jpg',cl1)

    def histogram_clip(self, img, clip_lo, clip_hi, depth, L_l=31, L_h= 248, eightBits = True):

        hist,bins = np.histogram(img.flatten(),depth,[0,depth-1])
        # print('hist:')
        # print(hist)

        # print("bins:")
        # print(bins)
        
        cdf = hist.cumsum()
        cdf_n = cdf/ cdf.max()

        # print("cdf_n:")
        # print(cdf_n)

        # print("cdf shape:",np.shape(cdf_n))

        # print("cdf_n lo and hi:",cdf_n.min(), cdf_n.max())
        # print("clip limits:",clip_lo, clip_hi)

        # rms = np.std(hist)
        # print('rms hist:',rms)

        # hist_max = np.median(hist)
        # print('hist_max:',hist_max)

        # index_lo = np.argmax(hist+2.0*rms )
        # index_lo = np.argmax(hist_max )
        index_lo = np.argmax(cdf_n > clip_lo)
        index_hi = np.argmax(cdf_n > clip_hi)

        # print("index_lo and hi:",index_lo,index_hi)

        # create the lookup table:
        if eightBits == True:
            lut = np.zeros(depth,np.uint8)
        else:
            lut = np.zeros(depth,np.uint16)

        if(index_hi == index_lo):
            m = L_h - L_l
        else:
            m = (L_h - L_l)/(index_hi - index_lo)

        # m = 0.8 * m

        # m = 2.0*m

        b = L_l - m*index_lo

        # print('m,b:',m,b)
        # print('depth:',depth)

        for i in np.arange(0,depth):
            if i < index_lo:
                lut[i] = L_l
            elif i > index_hi:
                lut[i] = depth-1
                # lut[i] = L_h
            else:
                if eightBits == True:
                    lut[i] = (m*i + b).astype(np.uint8)
                    if lut[i] < L_l:
                        lut[i] = L_l
                else:
                    # lut[i] = (m*i + b).astype(np.uint16)
                    lut[i] = (m*i + b).astype(np.uint16)
                    if lut[i] < L_l:
                        lut[i] = L_l

        # print('lut[0]',lut[0])
        # print('lut:')
        # print(lut)

        return lut

    def apply_lut(self,img,lut):
        img_out = np.empty_like(img)
        img_out[:,:] = lut[img[:,:]]
        return img_out

    def getXYCoordsForFITS(self, astro_file_name, fitsFileName):
        # gets the list of (x,y) coordinates for each observation in the FITS fileName located in the directory astro_file_name:
        fits_file_names_list =self.astroDict[astro_file_name]['header']['input_file']
        indices_list = [idx for idx,val in enumerate(fits_file_names_list) if fitsFileName in val]

        #get the number of observations (i.e. detected objects) list:
        observations_list = self.astroDict[astro_file_name]['header']['objects']

        # get the list of observed detections:
        detections_list =  self.astroDict[astro_file_name]['detections']['object_ast']
        # get the x,y coordinates lists:
        # x_coord_list =self.astroDict[astro_file_name]['object_astrometry']['x_center']
        # y_coord_list =self.astroDict[astro_file_name]['object_astrometry']['y_center']

        #get x,y values and save in a list of tuples:
        xy_coords = []
        for idx in indices_list:
            num_obs = int(observations_list[idx])
            if num_obs > 0:
                obj_ast_list = detections_list[idx]
                # iterate through the list items:
                for obj_ast_list_item in obj_ast_list: 
                    x = float(obj_ast_list_item['x_center'])
                    y = float(obj_ast_list_item['y_center'])
                    xy_coords.append((x,y))
        return xy_coords

    def annotateImage(self,img,class_name,xy_list,box_width,box_height):
        # img_annot_gray = np.copy(img)
        img_annot_rgb = np.copy(img)

        # convert to a color image:
        # img_annot_rgb = cv2.cvtColor(img_annot_gray, cv2.COLOR_GRAY2RGB)

        #NB" it's a 16 bit image!
        green = ((0,255*256,0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # [h,w] = np.shape(img_annot_rgb)
        h, w, channels = img_annot_rgb.shape

        for t in xy_list:
            x_0 = int(t[0])
            y_0 = int(t[1])
            left = int(x_0 - box_width/2)
            right = int(x_0 + box_width/2)
            top = int(y_0 - box_height/2)
            bot = int(y_0 + box_height/2)
            if(left < 0):
                left = 0
            if(top < 0):
                top = 0
            if(right > w):
                right = w
            if(bot > h):
                bot = h
            # print(class_name,h,w,x_0,y_0,left,right,top,bot)
            cv2.rectangle(img_annot_rgb, (left, top), (right, bot), green, 1)
            cv2.putText(img_annot_rgb, class_name, (left, top-4), font, 0.5, green, 2);
        return img_annot_rgb

    def annotateImage2(self,img,roi):

        class_name = 'sat'
        
        img_annot_rgb = np.copy(img)

        #NB" it's a 8 bit image!
        green = ((0,255,0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # [h,w] = np.shape(img_annot_rgb)
        h, w, channels = img_annot_rgb.shape

        x_0 = roi['x_center']
        y_0 = roi['y_center']

        n = len(roi['label'])

        for i in range(n):
            left = int((x_0[i] - roi['bbox_width'][i]/2)*w)
            right = int((x_0[i] + roi['bbox_width'][i]/2)*w)
            top = int((y_0[i] - roi['bbox_height'][i]/2)*h)
            bot = int((y_0[i] + roi['bbox_height'][i]/2)*h)
            if(left < 0):
                left = 0
            if(top < 0):
                top = 0
            if(right > w):
                right = w
            if(bot > h):
                bot = h
            # print(class_name,h,w,x_0,y_0,left,right,top,bot)
            cv2.rectangle(img_annot_rgb, (left, top), (right, bot), green, 1)
            cv2.putText(img_annot_rgb, class_name, (left, top-4), font, 0.5, green, 2);
        return img_annot_rgb

    def addtoAnnotatedImage(self,path2PngAnnot,class_name,x_center,y_center,box_width,box_height):

        # print(path2PngAnnot,class_name,x_center,y_center,box_width,box_height)

        img_annot_rgb = cv2.imread(path2PngAnnot,cv2.IMREAD_COLOR)
        green = ((0,255*256,0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        h, w, channels = img_annot_rgb.shape

        x_0 = int(x_center)
        y_0 = int(y_center)
        left = int(x_0 - box_width/2)
        right = int(x_0 + box_width/2)
        top = int(y_0 - box_height/2)
        bot = int(y_0 + box_height/2)
        if(left < 0):
            left = 0
        if(top < 0):
            top = 0
        if(right > w):
            right = w
        if(bot > h):
            bot = h
        # print(class_name,h,w,x_0,y_0,left,right,top,bot)
        cv2.rectangle(img_annot_rgb, (left, top), (right, bot), green, 1)
        cv2.putText(img_annot_rgb, class_name, (left, top-4), font, 0.5, green, 2);
        cv2.imwrite(path2PngAnnot,img_annot_rgb)

    def scale_image(self,image, scale):
        height, width = image.shape[:2]
        image_resized = cv2.resize(image,(scale*width, scale*height), interpolation = cv2.INTER_CUBIC)
        # print('scale_image:',image.shape[:2])
        # print('scale_image:',image_resized.shape[:2])
        return image_resized

    def extract_roi(self,image,ROI,IMAGE_WIDTH,IMAGE_HEIGHT):
        offset = 0.5
        x = ROI['x']
        y = ROI['y']
        w = ROI['w']
        h = ROI['h']
        x0 = int(x-w/2)
        x1 = int(x+w/2)
        y0 = int(y-h/2)
        y1 = int(y+h/2)

        image_roi = image[y0:y1, x0:x1]
        # print('extract_roi x0,x1,y0,y1,x,y,w,h:',x0,x1,y0,y1,x,y,w,h)
        # print('extract_roi:',image.shape[:2])
        # print('extract_roi:',image_roi.shape[:2])
        return image_roi

    def getStats(self,astro_file_name):
        # count the number of satellite detections per FITS image file: return a dictionary with this count for statistical analysis
        fits_file_names_list =self.astroDict[astro_file_name]['header']['input_file']
        indices_list = [idx for idx,val in enumerate(fits_file_names_list)]

        #get the number of observations (i.e. detected objects) list:
        observations_list = self.astroDict[astro_file_name]['header']['objects']

        # save the number of detections per image in a dictionary:
        hist_dict = defaultdict(int)
        for idx in indices_list:
            num_obs = int(observations_list[idx])
            hist_dict[num_obs] += 1
        return sorted(hist_dict.items(), key=itemgetter(0))

    def getNumObs(self,astro_file_name,fitsFileName):
        # gets the number of observations in the FITS fileName located in the directory astro_file_name:
        fits_file_names_list =self.astroDict[astro_file_name]['header']['input_file']
        indices_list = [idx for idx,val in enumerate(fits_file_names_list) if fitsFileName in val]

        #get the number of observations (i.e. detected objects) list:
        observations_list = self.astroDict[astro_file_name]['header']['objects']

        # get number of observations:
        num_obs = 0
        for idx in indices_list:
            num_obs += int(observations_list[idx])
        return num_obs

    def getNumObsPerFile(self,astro_file_name):
        # creates a dictionary of number of observations per filename:
        fits_file_names_list =self.astroDict[astro_file_name]['header']['input_file']
        indices_list = [idx for idx,val in enumerate(fits_file_names_list)]

        #get the number of observations (i.e. detected objects) list:
        observations_list = self.astroDict[astro_file_name]['header']['objects']

        obs_dict = {}

        for idx in indices_list:
            file_name = fits_file_names_list[idx].split('/')
            file_name = file_name[len(file_name)-1]
            file_name = file_name.replace('.fits','')
            # print(file_name)
            # sys.exit()
            obs_dict[file_name] = observations_list[idx]

        return obs_dict

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or type(resFile) == unicode:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]
        print('DONE (t={:0.2f}s)'.format(time.time()- tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir = None, imgIds = [] ):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m