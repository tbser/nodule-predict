#!usr/bin/env python
# -*-coding:utf-8-*-
import os
import sys
import glob
import pandas


"""filefolder/csv文件s  对这些csv文件进行操作 count每个csv下的条目"""
def count_pos(src_dir):
    count = 0
    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        for i in df_annos.iterrows():
            count += 1

    return count


if __name__ == '__main__':
    # 文件夹路径作为命令行参数
    if len(sys.argv) < 2:
        print('Usage: {} <source directory> )'.format(sys.argv[0]))
        sys.exit(1)

    src_dir = sys.argv[1]
    print("src_dir:", src_dir)
    if not os.path.isdir(src_dir):
        print('Invalid source directory: {}'.format(src_dir))
        sys.exit(2)

    count = count_pos(src_dir=src_dir)
    print("count:", count)

