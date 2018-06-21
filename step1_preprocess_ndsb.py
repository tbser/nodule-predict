import settings
import helpers
import glob
import os
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import scipy.misc
import dicom  # pip install pydicom
import numpy
import math
import timeprofile
from timeprofile import calltimeprofile, print_prof_data
import logging.config
#from logging import handlers
import sys
from multiprocessing import Pool

#logfile = os.getcwd() + "/step1-ndsb.log"
#logging.basicConfig(filename=logfile, level=logging.INFO)
logging.config.fileConfig("logging.conf")
settings.log = logging.getLogger('stpe1_ndsb')
#stdout= logging.StreamHandler(sys.stdout)
#log.addHandler(stdout)

def load_patient(src_dir):
    slices = [dicom.read_file(src_dir + '/' + s) for s in os.listdir(src_dir)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def load_patient2(src_dir):
    lstFilesDCM = []
    for dirname, subdirlist, filelist in os.walk(src_dir):
        for filename in filelist:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirname, filename))
    slices = [dicom.read_file(s) for s in lstFilesDCM]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = numpy.stack([s.pixel_array for s in slices])
    image = image.astype(numpy.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(numpy.float64)
            image[slice_number] = image[slice_number].astype(numpy.int16)
        image[slice_number] += numpy.int16(intercept)

    return numpy.array(image, dtype=numpy.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = numpy.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = numpy.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing

def cv_flip(img,cols,rows,degree):
    M = cv2.getRotationMatrix2D((cols / 2, rows /2), degree, 1.0)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def extract_dicom_images_patient(src_dir):
    target_dir = settings.NDSB3_EXTRACTED_IMAGE_DIR
    #print("Dir: ", src_dir)
    settings.log.info("Patient: {0}".format(src_dir))
    dir_path = settings.NDSB3_RAW_SRC_DIR + src_dir + "/"
    patient_id = src_dir
    slices = load_patient(dir_path)
    settings.log.info("Slice number: {0} \t thickness - {1} \t x and y spacing - {2}".format(len(slices), slices[0].SliceThickness, slices[0].PixelSpacing))
    #print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
    settings.log.info("Orientation: {0}".format(slices[0].ImageOrientationPatient))
    #print("Orientation: ", slices[0].ImageOrientationPatient)
    #assert slices[0].ImageOrientationPatient == [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
    cos_value = (slices[0].ImageOrientationPatient[0])
    cos_degree = round(math.degrees(math.acos(cos_value)),2)
    
    pixels = get_pixels_hu(slices)
    image = pixels
    settings.log.info("image shape: {0}".format(image.shape))

    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    settings.log.info("Invert order: {0} - {1:.6f}, {2:.6f}".format(invert_order, slices[1].ImagePositionPatient[2], slices[0].ImagePositionPatient[2]))
    #print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",", slices[0].ImagePositionPatient[2])

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    image = helpers.rescale_patient_images(image, pixel_spacing, settings.TARGET_VOXEL_MM, verbose=True)
    if not invert_order:
        image = numpy.flipud(image)

    for i in range(image.shape[0]):
        patient_dir = target_dir + patient_id + "/"
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        img_path = patient_dir + "img_" + str(i).rjust(4, '0') + "_i.png"
        org_img = image[i]
        # if there exists slope,rotation image with corresponding degree
        if cos_degree>0.0:
            org_img = cv_flip(org_img,org_img.shape[1],org_img.shape[0],cos_degree)
        img, mask = helpers.get_segmented_lungs(org_img.copy())
        org_img = helpers.normalize_hu(org_img)
        cv2.imwrite(img_path, org_img * 255)
        cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)

@calltimeprofile
def extract_dicom_images(clean_targetdir_first=False, only_patient_id=None):
    settings.log.info("Extracting images")

    target_dir = settings.NDSB3_EXTRACTED_IMAGE_DIR
    if clean_targetdir_first and only_patient_id is not None:
        settings.log.info("Cleaning target dir")
        for file_path in glob.glob(target_dir + "*.*"):
            os.remove(file_path)

    if only_patient_id is None:
        dirs = os.listdir(settings.NDSB3_RAW_SRC_DIR)
        if True:
            pool = Pool(8)
            pool.map(extract_dicom_images_patient, dirs)
        else:
            for dir_entry in dirs:
                extract_dicom_images_patient(dir_entry)
    else:
        extract_dicom_images_patient(only_patient_id)


if __name__ == '__main__':
    extract_dicom_images(clean_targetdir_first=False, only_patient_id=None)
    print_prof_data()
    logging.shutdown()