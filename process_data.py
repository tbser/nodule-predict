import os
import pandas as pd
import settings
import helpers
import dicom

mylogger = helpers.getlogger('process_data.log')

CUBE_SIZE = 32

def merge_nodule_detector_results(patient_dir, result_dir, logger = mylogger):
    if not os.path.exists(patient_dir) or not os.path.exists(result_dir):
        return
    dst_dir = result_dir + "merge/"    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for patient_id in os.listdir(patient_dir):
        merge_lines = []
        #merge_index = 0
        for model_name in os.listdir(result_dir):
            if model_name != "merge":
                csv_path = result_dir + model_name + "/" + patient_id + ".csv"
                if not os.path.exists(csv_path):
                    logger.info("{0} does not exist.".format(csv_path))
                else:
                    result = pd.read_csv(csv_path)
                    for ind, row in result.iterrows():
                        row["anno_index"] = model_name + "_" + str(ind)
                        merge_lines.append(row) 
        if (len(merge_lines) > 0):
            df = pd.DataFrame(merge_lines, columns = result.columns)
            df.to_csv(dst_dir + patient_id + ".csv", index=False)       
    return

def transform_prediction_results(dicom_src_dr, predict_result_src_dir, transform_dir, logger = mylogger):
    #src_dir_result = settings.NDSB3_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"
    #src_dir_dcm = settings.NDSB3_RAW_SRC_DIR
    #trans_dir = settings.NDSB3_TRANSFORM_DIR
    #if not os.path.exists(trans_dir):
    #    os.mkdir(trans_dir)
    #dst_dir = transform_dir + "predictions" + str(int(magnification * 10)) + ext_name + "/"
    if not os.path.exists(transform_dir):
        os.makedirs(transform_dir)
    logger.info("===========transform: {0}".format(predict_result_src_dir))

    for patient in os.listdir(dicom_src_dir):
        if patient.startswith('.'):   # 忽略mac中的隐藏文件
            continue
        logger.info("patient_id: {}".format(patient))
        lstFilesDCM = []
        for dirname, subdirlist, filelist in os.walk(dicom_src_dir + patient + "/"):
            for filename in filelist:
                if ".dcm" in filename.lower():
                    lstFilesDCM.append(os.path.join(dirname, filename))
        slices = [dicom.read_file(s) for s in lstFilesDCM]
        dcm_x = slices[0].pixel_array.shape[0]
        dcm_y = slices[0].pixel_array.shape[1]
        dcm_z = len(slices)
        spacing_y = slices[0].PixelSpacing[0]
        spacing_x = slices[0].PixelSpacing[1]
        thickness = slices[0].SliceThickness

        csv_file = predict_result_src_dir + patient + '.csv'
        if not os.path.exists(csv_file):
            logger.info("{} does not exist. Skip to next dicom series!".format(csv_file))
            continue
        prediction = pd.read_csv(csv_file)
        if len(prediction) == 0:
            continue
        trans_line = []
        for index, row in prediction.iterrows():
            trans_x = int(row["coord_x"] * dcm_x)
            trans_y = int(row["coord_y"] * dcm_y)
            trans_z = int(row["coord_z"] * dcm_z)
            x_diameter = int(CUBE_SIZE/spacing_x)
            y_diameter = int(CUBE_SIZE/spacing_y)
            z_diameter = int(CUBE_SIZE/thickness)
            logger.info("transform_result:({0}, {1}, {2})".format(trans_x, trans_y, trans_z))
            trans_line.append([row["anno_index"], trans_x, trans_y, trans_z, x_diameter,y_diameter, z_diameter, row["nodule_chance"], row["diameter_mm"]])

        trans_result = pd.DataFrame(trans_line, columns=["anno_index", "trans_x", "trans_y", "trans_z", "x_diameter", "y_diameter", "z_diameter", "nodule_chance", "maligncy"])
        trans_result.to_csv(transform_dir + patient + "_trans.csv", index=False)

if __name__ == "__main__":
    #merge_nodule_detector_results(settings.HOSPITAL_EXTRACTED_IMAGE_DIR, settings.HOSPITAL_NODULE_DETECTION_DIR)
    dicom_src_dir = "/opt/data/deeplearning/lung-data/第一批/肺部小结节薄层CT数据/"

    for magnification in [1, 1.5, 2]:
        ext_name = "_luna16_fs"
        predict_result_src_dir = settings.HOSPITAL_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"
        transform_dir = settings.HOSPITAL_TRANSFORM_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"
        transform_prediction_results(dicom_src_dir, predict_result_src_dir, transform_dir)
        
        ext_name = "_luna_posnegndsb_v1"
        predict_result_src_dir = settings.HOSPITAL_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"
        transform_dir = settings.HOSPITAL_TRANSFORM_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"

        transform_prediction_results(dicom_src_dir, predict_result_src_dir, transform_dir)
        
        ext_name = "_luna_posnegndsb_v2"
        predict_result_src_dir = settings.HOSPITAL_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"
        transform_dir = settings.HOSPITAL_TRANSFORM_DIR + "predictions" + str(int(magnification * 10)) + ext_name + "/"
