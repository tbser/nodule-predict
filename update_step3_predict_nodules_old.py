import os
os.environ['PYTHONHASHSEED'] = '0'
import settings
import helpers
import glob
import random
import pandas
import ntpath
import numpy
from keras import backend as K
import math
import shutil

# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
# numpy.random.seed(42)

# The below is necessary for starting core Python generated random numbers in a well-defined state.
random.seed(2)

# limit memory usage..
import tensorflow as tf

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

# tf.set_random_seed(1234)
from keras.backend.tensorflow_backend import set_session
import step2_train_nodule_detector

# logger = helpers.getlogger('update_step3_predict_nodules.log')
logger = helpers.getlogger(os.path.splitext(os.path.basename(__file__))[0] + '.log')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")

CUBE_SIZE = step2_train_nodule_detector.CUBE_SIZE
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
NEGS_PER_POS = 20
P_TH = 0.6

PREDICT_STEP = 12
USE_DROPOUT = False


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def filter_patient_nodules_predictions(df_nodule_predictions: pandas.DataFrame, patient_id, view_size,
                                       data_source="ndsb3", luna16=False):
    if data_source == "ndsb3":
        src_dir = settings.NDSB3_EXTRACTED_IMAGE_DIR
    elif data_source == "luna":
        src_dir = settings.LUNA_16_TRAIN_DIR2D2
    else:
        src_dir = settings.HOSPITAL_EXTRACTED_IMAGE_DIR

    patient_mask = helpers.load_patient_images(patient_id, src_dir, "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nodule_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y + view_size, start_x:start_x + view_size]
            if img_roi.sum() > 255:  # more than 1 pixel of mask.
                logger.info("More than 1 pixel of mask. nodule_in_mask is true")
                nodule_in_mask = True

        if not nodule_in_mask:
            logger.info("Nodule not in mask: {0} {1} {2}".format(center_x, center_y, center_z))
            if mal_score > 0:
                mal_score *= -1
            df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                logger.info("Z < 30: {0} center z: {1}  y_perc: {2} ".format(patient_id, center_z, y_perc))
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score

            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                logger.info(
                    "SUSPICIOUS FALSEPOSITIVE: {0}  center z: {1}  y_perc: {2}".format(patient_id, center_z, y_perc))

            if center_z < 50 and y_perc < 0.30:
                logger.info(
                    "SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: {0} center z: {1} y_perc: {2}".format(patient_id, center_z,
                                                                                                  y_perc))

    df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions


# no usage
def filter_nodule_predictions(only_patient_id=None):
    src_dir = settings.NDSB3_NODULE_DETECTION_DIR
    for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
        file_name = ntpath.basename(csv_path)
        patient_id = file_name.replace(".csv", "")
        logger.info("csv_index {0} : patient_id {1}".format(csv_index, patient_id))
        if only_patient_id is not None and patient_id != only_patient_id:
            continue
        df_nodule_predictions = pandas.read_csv(csv_path)
        filter_patient_nodules_predictions(df_nodule_predictions, patient_id, CUBE_SIZE)
        df_nodule_predictions.to_csv(csv_path, index=False)


# no usage
def make_negative_train_data_based_on_predicted_luna_nodules():
    src_dir = settings.LUNA_NODULE_DETECTION_DIR
    pos_labels_dir = settings.LUNA_NODULE_LABELS_DIR
    keep_dist = CUBE_SIZE + CUBE_SIZE / 2
    total_false_pos = 0
    for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
        file_name = ntpath.basename(csv_path)
        patient_id = file_name.replace(".csv", "")
        # if not "273525289046256012743471155680" in patient_id:
        #     continue
        df_nodule_predictions = pandas.read_csv(csv_path)
        pos_annos_manual = None
        manual_path = settings.MANUAL_ANNOTATIONS_LABELS_DIR + patient_id + ".csv"
        if os.path.exists(manual_path):
            pos_annos_manual = pandas.read_csv(manual_path)

        filter_patient_nodules_predictions(df_nodule_predictions, patient_id, CUBE_SIZE, luna16=True)
        pos_labels = pandas.read_csv(pos_labels_dir + patient_id + "_annos_pos_lidc.csv")
        logger.info("csv_index {0} : patient_id {1} , pos {2}".format(csv_index, patient_id, len(pos_labels)))
        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA_16_TRAIN_DIR2D2, "*_m.png")
        for nod_pred_index, nod_pred_row in df_nodule_predictions.iterrows():
            if nod_pred_row["diameter_mm"] < 0:
                continue
            nx, ny, nz = helpers.percentage_to_pixels(nod_pred_row["coord_x"], nod_pred_row["coord_y"],
                                                      nod_pred_row["coord_z"], patient_imgs)
            diam_mm = nod_pred_row["diameter_mm"]
            for label_index, label_row in pos_labels.iterrows():
                px, py, pz = helpers.percentage_to_pixels(label_row["coord_x"], label_row["coord_y"],
                                                          label_row["coord_z"], patient_imgs)
                dist = math.sqrt(math.pow(nx - px, 2) + math.pow(ny - py, 2) + math.pow(nz - pz, 2))
                if dist < keep_dist:
                    if diam_mm >= 0:
                        diam_mm *= -1
                    df_nodule_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                    break

            if pos_annos_manual is not None:
                for index, label_row in pos_annos_manual.iterrows():
                    px, py, pz = helpers.percentage_to_pixels(label_row["x"], label_row["y"], label_row["z"],
                                                              patient_imgs)
                    diameter = label_row["d"] * patient_imgs[0].shape[1]
                    # print((pos_coord_x, pos_coord_y, pos_coord_z))
                    # print(center_float_rescaled)
                    dist = math.sqrt(math.pow(px - nx, 2) + math.pow(py - ny, 2) + math.pow(pz - nz, 2))
                    if dist < (diameter + 72):  # make sure we have a big margin
                        if diam_mm >= 0:
                            diam_mm *= -1
                        df_nodule_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                        logger.info("#Too close: {0} {1} {2}".format(nx, ny, nz))
                        break

        df_nodule_predictions.to_csv(csv_path, index=False)
        df_nodule_predictions = df_nodule_predictions[df_nodule_predictions["diameter_mm"] >= 0]
        df_nodule_predictions.to_csv(pos_labels_dir + patient_id + "_candidates_falsepos.csv", index=False)
        total_false_pos += len(df_nodule_predictions)
    logger.info("Total false pos: {0}".format(total_false_pos))


# update
def data_generator(test_files, data_source):
    img_list = []
    # while True:
    CROP_SIZE = CUBE_SIZE
    for test_idx, test_item in enumerate(test_files):
        file_name = ntpath.basename(test_item)
        parts = file_name.split('_')
        # logger.info("data_generator:file_name {0}".format(file_name))

        # if parts[0] == "ndsb3manual" or parts[0] == "hostpitalmanual":
        #     patient_id = parts[1]
        # else:
        #     patient_id = parts[0]

        if data_source == "testdata_neg" and parts[0] != "ndsb3manual":  # 除了ndsb3manual  其他neg都是6*8
            """6*8 情形"""
            # logger.info("situation 6*8")
            cube_image = helpers.load_cube_img(test_item, 6, 8, 48)
            # logger.info("cube image: {0}".format(cube_image))
            wiggle = 48 - CROP_SIZE - 1
            indent_x = 0
            indent_y = 0
            indent_z = 0
            if wiggle > 0:
                indent_x = random.randint(0, wiggle)
                indent_y = random.randint(0, wiggle)
                indent_z = random.randint(0, wiggle)
            cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                         indent_x:indent_x + CROP_SIZE]
            # logger.info("cube_image with indent_x(random.randint(0,wiggle)): {0}".format(cube_image))
            if CROP_SIZE != CUBE_SIZE:
                cube_image = helpers.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
            assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

        else:  # pos的都是8*8的  ndsb3manual的neg也是8*8的
            """8*8 情形"""
            # logger.info("situation 8*8")
            cube_image = helpers.load_cube_img(test_item, 8, 8, 64)
            # logger.info("cube image: {0}".format(cube_image))
            current_cube_size = cube_image.shape[0]
            wiggle_indent = 0
            wiggle = current_cube_size - CROP_SIZE - 1

            if wiggle > (CROP_SIZE / 2):
                wiggle_indent = CROP_SIZE / 4
                wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1

            indent_x = wiggle_indent + random.randint(0, wiggle)
            indent_y = wiggle_indent + random.randint(0, wiggle)
            indent_z = wiggle_indent + random.randint(0, wiggle)
            
            indent_x = int(indent_x)
            indent_y = int(indent_y)
            indent_z = int(indent_z)
            cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE,
                         indent_x:indent_x + CROP_SIZE]
            # logger.info("cube_image with indent_x(random.randint(0,wiggle)):{0}".format(cube_image))

            if CROP_SIZE != CUBE_SIZE:
                cube_image = helpers.rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
            assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

        img3d = prepare_image_for_net3D(cube_image)
        img_list.append((img3d, file_name))
        # logger.info("img_list:{0}".format(img_list))
        # batch_idx += 1
        # if batch_idx >= batch_size:
        #     x = numpy.vstack(img_list)
        #     yield x
        #     img_list = []
        #     batch_idx = 0

    return img_list


def filter_wrong_predict_file(category, file):
    if not os.path.exists(category):
        os.makedirs(category)
    shutil.copy(file, category)


# predict输入把目录改成区域
def predict_area(model_path, data_source="testdata_neg", flip=False, ext_name=""):
    logger.info("Predict cubes with model {0}, data_source {1} ".format(model_path, data_source))
    if data_source == "testdata_neg":
        src_dir = settings.SEPARATE_TESTDATA_NEG_DIR
        dst_dir = settings.PREDICT_TESTDATA_NEG_DIR
    else:
        src_dir = settings.SEPARATE_TESTDATA_POS_DIR
        dst_dir = settings.PREDICT_TESTDATA_POS_DIR

    holdout_ext = ""
    # if holdout_no is not None:
    #     holdout_ext = "_h" + str(holdout_no) if holdout_no >= 0 else ""
    flip_ext = ""
    if flip:
        flip_ext = "_flip"

    dst_dir += "predictions" + holdout_ext + flip_ext + "_" + ext_name + "/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    model = step2_train_nodule_detector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1),
                                                load_weight_path=model_path)
    logger.info("=====model has been loaded=====")
    test_files = glob.glob(src_dir + "*.png")
    # helpers.load_cube_img(test_item, 6, 8, 48)
    # helpers.load_cube_img(test_item, 8, 8, 64)
    img_list = data_generator(test_files, data_source)
    logger.info("img_list(data_generator) is ok.")
    # batch_size = 128
    batch_size = 1  # for test
    batch_list = []
    batch_list_loc = []
    annotation_index = 0
    area_predictions_csv = []
    true_positive, false_negative = 0, 0
    true_negative, false_positive = 0, 0
    count = 0

    for item in img_list:
        cube_img = item[0]
        file_name = item[1]
        parts = file_name.split('_')
        if parts[0] == "ndsb3manual" or parts[0] == "hostpitalmanual":
            patient_id = parts[1]
        else:
            patient_id = parts[0]
        logger.info("{0} - patient_id {1}".format(count, patient_id))
        # logger.info("the shape of cube image: {0}".format(numpy.array(cube_img).shape)) # (1, 32, 32, 32, 1)
        count += 1
        batch_list.append(cube_img)
        batch_list_loc.append(file_name)
        # logger.info("batch list: {0}".format(batch_list))
        # logger.info("the shape of batch list: {0}".format(numpy.array(batch_list).shape)) # (1, 1, 32, 32, 32, 1)
        # logger.info("batch list loc: {0}".format(batch_list_loc))

        if len(batch_list) % batch_size == 0:
            batch_data = numpy.vstack(batch_list)
            p = model.predict(batch_data, batch_size=batch_size)
            logger.info("the prediction result p: {0}".format(p))
            logger.info("the shape of p:{0}".format(numpy.array(p).shape))
            logger.info("=====the length of p[0]:{0}".format(len(p[0])))
            for i in range(len(p[0])):
                file_name = batch_list_loc[i]
                csv_target_path = dst_dir + os.path.splitext(file_name)[0] + ".csv"
                nodule_chance = p[0][i][0]
                diameter_mm = round(p[1][i][0], 4)
                nodule_chance = round(nodule_chance, 4)
                logger.info("csv_target_path:{0}".format(csv_target_path))
                logger.info("nodule chance:{0}".format(nodule_chance))
                logger.info("Cube diameter_mm {0} ".format(diameter_mm))

                if data_source == "testdata_pos":
                    if nodule_chance > P_TH:
                        true_positive += 1
                        result = "true positive"
                    else:
                        false_negative += 1
                        result = "false negative"
                        filter_wrong_predict_file(settings.WRONG_PREDICTION_FN, src_dir + file_name)
                else:
                    if nodule_chance > P_TH:
                        false_positive += 1
                        result = "false positive"
                        filter_wrong_predict_file(settings.WRONG_PREDICTION_FP, src_dir + file_name)
                    else:
                        true_negative += 1
                        result = "true negative"

                area_predictions_csv_line = [annotation_index, nodule_chance, diameter_mm, result]
                area_predictions_csv.append(area_predictions_csv_line)
                logger.info("the shape of area_predictions_csv:{0}".format(numpy.array(area_predictions_csv).shape))
                annotation_index += 1
                # logger.info("pandas.dataframe begginning...")
                df = pandas.DataFrame(area_predictions_csv, columns=["anno_index", "nodule_chance", "diameter_mm", "result"])
                logger.info("pandas.dataframe done...")
                df.to_csv(csv_target_path, index=False)

                annotation_index = 0
                area_predictions_csv = []
                logger.info("area_predictions_csv has been cleared.")

            batch_list = []
            batch_list_loc = []
            # count = 0

    if data_source == "testdata_pos":
        return true_positive, false_negative
    else:
        return false_positive, true_negative


if __name__ == "__main__":
    try:
        if True:
            true_positive, false_negative = predict_area("/opt/data/deeplearning/train_result/3DCNN/models/model_luna16_full__fs_best.hd5", "testdata_pos", flip=False, ext_name="luna16_fs")
            false_positive, true_negative = predict_area("/opt/data/deeplearning/train_result/3DCNN/models/model_luna16_full__fs_best.hd5", "testdata_neg", flip=False, ext_name="luna16_fs")

            true_positive_rate = true_positive / (true_positive + false_negative)
            false_positive_rate = false_positive / (false_positive + true_negative)
            true_negative_rate = true_negative / (false_positive + true_negative)

            accuracy = (true_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)
            # Precision = 提取出的正确信息条数 / 提取出的信息条数 = TP / (TP + FP)
            # Recall = 提取出的正确信息条数 /  样本中的信息条数 = TP / (TP + FN)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)

            logger.info("true positive: {0} positive data are predicted as positive.".format(true_positive))
            logger.info("false negative: {0} positive data are predicted as negative.".format(false_negative))
            logger.info("false positive: {0} negative data are predicted as positive.".format(false_positive))
            logger.info("true negative: {0} negative data are predicted as negative.".format(true_negative))

            logger.info("FPR(false positive rate) = FP / (FP + TN) = {0}".format(false_positive_rate))
            logger.info("灵敏度sensitivity = TPR(true positive rate) = TP / (TP + FN) = {0}".format(true_positive_rate))
            logger.info("特异性specificity = TNR(True Negative Rate) = TN /(FP + TN) = 1 - FPR = {0}".format(true_negative_rate))
            logger.info("accuracy = (TP + TN) / (Total population) = {0}".format(accuracy))
            logger.info("precision = TP / (TP + FP) = {0}".format(precision))
            logger.info("recall = TP / (TP + FN) = {0}".format(recall))

        # if True:
        #     for version in [2, 1]:
        #         for holdout in [0, 1]:
        #             predict_area("/opt/data/deeplearning/train_result/3DCNN_separate_train_data/models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5",
        #                     "testdata_pos", flip=False, ext_name="luna_posnegndsb_v" + str(version))
        #             predict_area("/opt/data/deeplearning/train_result/3DCNN_separate_train_data/models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5",
        #                     "testdata_neg", flip=False, ext_name="luna_posnegndsb_v" + str(version))
        #             if holdout == 0:
        #                 predict_area(
        #                         "/opt/data/deeplearning/train_result/3DCNN_separate_train_data/models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5",
        #                         "testdata_pos", flip=False, ext_name="luna_posnegndsb_v" + str(version))
        #                 predict_area(
        #                         "/opt/data/deeplearning/train_result/3DCNN_separate_train_data/models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5",
        #                         "testdata_neg", flip=False, ext_name="luna_posnegndsb_v" + str(version))

            logger.handlers.clear()

    except Exception as ex:
        logger.exception(ex)

