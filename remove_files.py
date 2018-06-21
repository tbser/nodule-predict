import os
import random

# 删除路径path下，任意n个文件。
def remove_file(path, n):
    if not os.path.isdir(path):   # 判断path是否为路径
        return
    
    file_list = []
    for root, dirs, files in os.walk(path):
        for i in files:
            file_path = os.path.join(root, i)
            # print(file_path)
            file_list.append(file_path)
    random.shuffle(file_list)
    for item in file_list[:n]:
        os.remove(item)
    print(len(file_list)) 

if __name__ == "__main__":
    remove_file("/home/meditool/lung_cancer_detecor/src/separate_testdata/neg_data/" , n=3000)

