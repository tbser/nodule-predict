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
logger = helpers.getlogger(os.path.splitext(os.path.basename(__file__))[0])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")

CUBE_SIZE = step2_train_nodule_detector.CUBE_SIZE
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
# NEGS_PER_POS = 20
# P_TH = 0.6
P_TH = 0.5

PREDICT_STEP = 12
USE_DROPOUT = False


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img

def filter_patient_nodules_predictions(df_nodule_predictions, patient_id, view_size, data_source="ndsb3", luna16=False):
# def filter_patient_nodules_predictions(df_nodule_predictions: pandas.DataFrame, patient_id, view_size, data_source="ndsb3", luna16=False):
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
        # logger.info("csv_index {0} : patient_id {1}".format(csv_index, patient_id))
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
        # logger.info("csv_index {0} : patient_id {1} , pos {2}".format(csv_index, patient_id, len(pos_labels)))
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
def data_generator(test_files):
    img_list = []
    # while True:
    CROP_SIZE = CUBE_SIZE
    for test_idx, test_item in enumerate(test_files):
        file_name = ntpath.basename(test_item)
        parts = file_name.split("_")
        pn = analysis_filename(file_name)[1]

        # logger.info("data_generator:file_name {0}".format(file_name))
        # logger.info("===pn:{0}".format(pn))

        if pn == "neg" and parts[0] != "ndsb3manual":  # 除了ndsb3manual  其他neg都是6*8
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

    return img_list


def filter_wrong_predict_file(file, category):
    if not os.path.exists(category):
        os.makedirs(category)
    shutil.copy(settings.SEPARATE_DATA_DIR + file, category)


def analysis_filename(file_name):
    # hostpitalmanual_CHEN-LEYAN_5_pos_0_4_1_pn.png
    # ndsb3manual_2d81a9e760a07b25903a8c4aeb444eca_1_pos_0_18_1_pn.png
    # 1.3.6.1.4.1.14519.5.2.1.6279.6001.254254303842550572473665729969_2945xpointx0_9_1_pos.png

    # 1.3.6.1.4.1.14519.5.2.1.6279.6001.315918264676377418120578391325_492_0_luna.png
    # 1.3.6.1.4.1.14519.5.2.1.6279.6001.707218743153927597786179232739_119_0_edge.png
    # ndsb3manual_6a7f1fd0196a97568f2991c885ac1c0b_1_neg_0_3_1_pn.png
    parts = os.path.splitext(file_name)[0].split("_")
    if parts[0] == "ndsb3manual" or parts[0] == "hostpitalmanual":
        patient_id = parts[1]
        pn = parts[3]
    else:
        patient_id = parts[0]
        if parts[-1] == "luna" or parts[-1] == "edge": pn = "neg"
        else:  pn = parts[-1]

    return patient_id, pn


def predict(model_path, img_list):
    model = step2_train_nodule_detector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)
    logger.info("=====model has been loaded=====")
    # batch_size = 128
    batch_size = 1  # for test
    batch_list = []
    batch_list_loc = []
    count = 0
    predictions = []

    for item in img_list:
        cube_img = item[0]
        file_name = item[1]
        patient_id = analysis_filename(file_name)[0]
        # logger.info("====={0} - patient_id {1}".format(count, patient_id))
        # logger.info("the shape of cube image: {0}".format(numpy.array(cube_img).shape)) # (1, 32, 32, 32, 1)
        count += 1
        batch_list.append(cube_img)
        batch_list_loc.append(file_name)
        # logger.info("batch list: {0}".format(batch_list))
        # logger.info("the shape of batch list: {0}".format(numpy.array(batch_list).shape)) # (1, 1, 32, 32, 32, 1)
        # logger.info("batch list loc: {0}".format(batch_list_loc))

        # if len(batch_list) % batch_size == 0:
        batch_data = numpy.vstack(batch_list)
        p = model.predict(batch_data, batch_size=batch_size)
        # logger.info("the prediction result p: {0}".format(p))
        # [array([[ 0.00064842]], dtype=float32), array([[  1.68593288e-05]], dtype=float32)]
        # logger.info("the shape of p:{0}".format(numpy.array(p).shape))  # (2, 1, 1)
        # logger.info("the length of p[0]:{0}".format(len(p[0])))  # 1

        # for i in range(len(p[0])):
        i = 0
        file_name = batch_list_loc[i]
        nodule_chance = p[0][i][0]
        diameter_mm = round(p[1][i][0], 4)
        nodule_chance = round(nodule_chance, 4)
        # logger.info("nodule chance:{0}, diameter_mm:{1}".format(nodule_chance, diameter_mm))
        item_prediction = [file_name, nodule_chance, diameter_mm]
        predictions.append(item_prediction)

        batch_list = []
        batch_list_loc = []
        # count = 0

    return predictions


def predict_area(model_path1, model_path2, flip=False, ext_name=""):
    logger.info("Predict cubes with model {0} and {1}".format(model_path1, model_path2))
    dst_dir = settings.PREDICT_DATA_DIR
    src_dir = settings.SEPARATE_DATA_DIR

    holdout_ext = ""
    # if holdout_no is not None:
    #     holdout_ext = "_h" + str(holdout_no) if holdout_no >= 0 else ""
    flip_ext = ""
    if flip:
        flip_ext = "_flip"

    dst_dir += "predictions" + holdout_ext + flip_ext + "_" + ext_name + "/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    test_files = glob.glob(src_dir + "*.png")
    img_list = data_generator(test_files)
    logger.info("img_list(data_generator) is ok.")

    predictions1 = predict(model_path1, img_list)
    predictions2 = predict(model_path2, img_list)
    # logger.info("predictions1[0][0] : {0}".format(predictions1[0][0]))
    # logger.info("predictions2[0][0] : {0}".format(predictions2[0][0]))
    true_positive, false_negative = 0, 0
    true_negative, false_positive = 0, 0
    area_predictions_csv = []

    for i in range(len(predictions1)):
        file_name1, nodule_chance1, diameter_mm1 = predictions1[i][0], predictions1[i][1], predictions1[i][2]
        file_name2, nodule_chance2, diameter_mm2 = predictions2[i][0], predictions2[i][1], predictions2[i][2]
        csv_target_path = dst_dir + os.path.splitext(file_name1)[0] + ".csv"
        # logger.info("csv_target_path:{0}".format(csv_target_path))
        pn = analysis_filename(file_name1)[1]

        # 1:
        # if nodule_chance1 == nodule_chance2:
        #     nodule_chance = nodule_chance1
        # else:
        #     nodule_chance = (nodule_chance1 + nodule_chance2) / 2
        # 2:
        # if nodule_chance1 == nodule_chance2:
        #     nodule_chance = nodule_chance1
        # elif abs(nodule_chance1-nodule_chance2)>=0.3:
        #     count+=1
        #     if nodule_chance1 > nodule_chance2:
        #         nodule_chance = nodule_chance1
        #     else:
        #         nodule_chance = nodule_chance2
        # else:
        #     nodule_chance = (nodule_chance1 + nodule_chance2) / 2
        # 3:
        if nodule_chance1 >= nodule_chance2:
            nodule_chance = nodule_chance1
        else:
            nodule_chance = nodule_chance2

        if pn == "pos":
            if nodule_chance >= P_TH:
                true_positive += 1
                result = "true positive"
            else:
                false_negative += 1
                result = "false negative"
                filter_wrong_predict_file(file_name1, settings.WRONG_PREDICTION_FN)
        else:
            if nodule_chance >= P_TH:
                false_positive += 1
                result = "false positive"
                filter_wrong_predict_file(file_name1, settings.WRONG_PREDICTION_FP)
            else:
                true_negative += 1
                result = "true negative"
        diameter_mm = (diameter_mm1+diameter_mm2)/2

        area_predictions_csv_line = [nodule_chance, diameter_mm, result]
        # logger.info("the shape of area_predictions_csv:{0}".format(numpy.array(area_predictions_csv).shape))
        # logger.info("pandas.dataframe begginning...")
        area_predictions_csv.append(area_predictions_csv_line)
        df = pandas.DataFrame(area_predictions_csv, columns=["nodule_chance", "diameter_mm", "result"])
        df.to_csv(csv_target_path, index=False)
        # logger.info("to_csv done...")
        area_predictions_csv = []

    return true_positive, false_negative, false_positive, true_negative


if __name__ == "__main__":
    try:
        if True:

            # predict_area("workdir_1_2_lecun_normal/model_luna16_full__fs_best.hd5", "workdir_1_2_lecun_aug2_luna_dropout_maxnorm/model_luna16_full__fs_best.hd5", flip=False, ext_name="luna16_fs")

            # /opt/data/deeplearning/train_result/3DCNN/
            true_positive, false_negative, false_positive, true_negative = predict_area("workdir_1_2_lecun_normal/model_luna16_full__fs_best.hd5", "workdir_1_2_lecun_aug2_luna_dropout_maxnorm/model_luna16_full__fs_best.hd5", flip=False, ext_name="luna16_fs")

            true_positive_rate = true_positive / (true_positive + false_negative)
            false_positive_rate = false_positive / (false_positive + true_negative)
            true_negative_rate = true_negative / (false_positive + true_negative)

            accuracy = (true_positive + true_negative) / (
            true_positive + false_negative + false_positive + true_negative)
            # Precision = 提取出的正确信息条数 / 提取出的信息条数 = TP / (TP + FP)
            # Recall = 提取出的正确信息条数 /  样本中的信息条数 = TP / (TP + FN)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1score = 2 * precision * recall / (precision + recall)
            logger.info("model:3DCNN")
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
            logger.info("F1Score = 2 * precision * recall / (precision + recall) = {0}".format(f1score))

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

