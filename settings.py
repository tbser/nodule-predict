#!usr/bin/env python
# -*-coding:utf-8-*-
import os

# COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", os.uname()[1])

global log
log = None
TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 1
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "/opt/data/deeplearning/"
BASE_DIR = "/opt/data/deeplearning/"
# EXTRA_DATA_DIR = BASE_DIR + "resources/"
WORKING_DIR = os.getcwd() + "/"
print("Current working directory is ", WORKING_DIR)
EXTRA_DATA_DIR = WORKING_DIR + "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "kaggle/dicom/"
# NDSB3_RAW_SRC_DIR = BASE_DIR + "lung-data/肺部小结节薄层CT数据/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna/"

NDSB3_EXTRACTED_IMAGE_DIR = WORKING_DIR + "ndsb3_extracted_images1/"
# NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "zhongshan_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = WORKING_DIR + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = WORKING_DIR + "ndsb3_nodule_predictions/"
HOSPITAL_DICOM_DIR = BASE_DIR + "lung-data/"
HOSPITAL_EXTRACTED_IMAGE_DIR = WORKING_DIR + "hospital_extracted_images/negative/"
HOSPITAL_NODULE_DETECTION_DIR = WORKING_DIR + "hospital_nodule_predictions/negative/"
HOSPITAL_TRANSFORM_DIR = WORKING_DIR + "hospital_nodule_predictions/positive/transform/"

# SEPARATE_TESTDATA_NEG_DIR = WORKING_DIR + "separate_testdata/neg_data/"
# SEPARATE_TESTDATA_POS_DIR = WORKING_DIR + "separate_testdata/pos_data/"

# PREDICT_TESTDATA_NEG_DIR = WORKING_DIR + "testdata_area_nodule_predictions/predict_neg_data/"
# PREDICT_TESTDATA_POS_DIR = WORKING_DIR + "testdata_area_nodule_predictions/predict_pos_data/"

WRONG_PREDICTION_FN = WORKING_DIR + "wrong_prediction/false_negative/"
WRONG_PREDICTION_FP = WORKING_DIR + "wrong_prediction/false_positive/"

PREDICT_DATA_DIR = WORKING_DIR + "testdata_area_nodule_predictions/"
# PREDICT_DATA_DIR = WORKING_DIR + "separate_testdata_more/"

SEPARATE_DATA_DIR = WORKING_DIR + "separate_testdata/"
