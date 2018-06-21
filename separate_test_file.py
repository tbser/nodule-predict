# -*- coding: utf-8 -*-

# test数据集 ：
# Luna   ndsb手工标注的   中山医院
import glob
import settings
import posixpath
import random
import numpy as np
import shutil
import os


def separate_train_test(test_percentage=10):
    """luna16   manual 37、lidc 5223: 都是pos    auto 522024: 都是neg: edge、luna、falsepos"""
    luna16_manual = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_manual/*.png")
    luna16_lidc = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_lidc/*.png")
    # neg_luna16_auto = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_auto/*.png")
    luna16_auto_edge = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_auto/*_edge.png")
    luna16_auto_luna = glob.glob(settings.WORKING_DIR + "generated_traindata/luna16_train_cubes_auto/*_luna.png")
   
    """医院 手工标注的 88  有pos 有neg"""
    hospital_manual = np.array(glob.glob(settings.WORKING_DIR + "generated_traindata/hospital_train_cubes_manual/*.png"))
    """ndsb 手工标注的 1387  有pos 有neg"""
    ndsb3_manual = np.array(glob.glob(settings.WORKING_DIR + "generated_traindata/ndsb3_train_cubes_manual/*.png"))

    # pos_samples 5260   neg_samples 514967
    pos_samples = luna16_manual + luna16_lidc
    neg_samples = luna16_auto_edge + luna16_auto_luna

    for files_path in [ndsb3_manual, hospital_manual]:
        for file_path in files_path:
            file_name = posixpath.basename(file_path)
            parts = file_name.split("_")
            # print(file_path)
            # for i in range(len(parts)):
            #     print(i, parts[i])
            if parts[3] == "pos":
                pos_samples.append(file_path)
            if parts[3] == "neg":
                neg_samples.append(file_path)

    random.shuffle(pos_samples)
    random.shuffle(neg_samples)
    # neg_samples_falsepos = []
    # for file_path in glob.glob(settings.BASE_DIR + "generated_traindata/luna16_train_cubes_auto/*_falsepos.png"):
    #     neg_samples_falsepos.append(file_path)
    # print("Falsepos LUNA count: ", len(neg_samples_falsepos))

    # print(len(pos_samples))   # 6537
    test_pos_count = int((len(pos_samples) * test_percentage) / 100)
    test_neg_count = int((len(neg_samples) * test_percentage) / 100)
    # print(test_pos_count, test_neg_count)  # 653 51516

    test_pos = pos_samples[:test_pos_count]
    test_neg = neg_samples[:test_neg_count]

    return test_pos, test_neg


if __name__ == '__main__':
    test_pos, test_neg = separate_train_test(test_percentage=10)
    # print(len(test_pos))
    dst_dir_test = settings.SEPARATE_DATA_DIR
    if not os.path.exists(dst_dir_test):
        os.makedirs(dst_dir_test)

    for path in test_pos:
        shutil.move(path, dst_dir_test)
    for path in test_neg:
        shutil.move(path, dst_dir_test)

