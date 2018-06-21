import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_result_diff_data(csv_file, model_name):
    train_result = pd.read_csv(csv_file)

    loss, out_class_binary_accuracy, out_class_loss, out_malignancy_loss = [], [], [], []
    val_loss, val_out_class_binary_accuracy, val_out_class_loss, val_out_malignancy_loss = [], [], [], []

    for index, row in train_result.iterrows():
        loss.append(row["loss"])
        out_class_binary_accuracy.append(row["out_class_binary_accuracy"])
        out_class_loss.append(row["out_class_loss"])
        out_malignancy_loss.append(row["out_malignancy_loss"])

        val_loss.append(row["val_loss"])
        val_out_class_binary_accuracy.append(row["val_out_class_binary_accuracy"])
        val_out_class_loss.append(row["val_out_class_loss"])
        val_out_malignancy_loss.append(row["val_out_malignancy_loss"])

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(2, 1, 1)
    ax.grid(True)
    ax.plot(out_class_binary_accuracy, color='orange', label='out_class_binary_accuracy', linewidth=1)
    ax.plot(loss, color='blue', label='loss', linewidth=1)
    # out_class_binary_crossentropy、out_class_loss 相等
    ax.plot(out_class_loss, color='green', label='out_class_loss', linewidth=1)
    # out_malignancy_mean_absolute_error、out_malignancy_loss 相等
    ax.plot(out_malignancy_loss, color='red', label='out_malignancy_loss', linewidth=1)

    ax.plot(val_out_class_binary_accuracy, color='orange', label='val_out_class_binary_accuracy', linewidth=1.6, linestyle='--')
    ax.plot(val_loss, color='blue', label='val_loss', linewidth=1.6, linestyle='--')
    ax.plot(val_out_class_loss, color='green', label='val_out_class_loss', linewidth=1.6, linestyle='--')
    ax.plot(val_out_malignancy_loss, color='red', label='val_out_malignancy_loss', linewidth=1.6, linestyle='--')

    plt.title(model_name)
    plt.xlabel('epoch')
    plt.ylabel('accuracy  &  loss')
    plt.xlim(0, 11)
    plt.ylim(0)
    plt.xticks(np.linspace(0, 11, 12))
    plt.yticks(np.linspace(0, 1, 10))

    plt.legend(['out_class_binary_accuracy', 'loss', 'out_class_loss', 'out_malignancy_loss',
                'val_out_class_binary_accuracy', 'val_loss', 'val_out_class_loss', 'val_out_malignancy_loss'],
               bbox_to_anchor=(0.5, -0.45), loc='upper center', ncol=2)


def plot_result_diff_archi(archi1_csv_file, model1_name, archi2_csv_file, model2_name):
    archi1_result = pd.read_csv(archi1_csv_file)
    archi2_result = pd.read_csv(archi2_csv_file)

    loss1, out_class_binary_accuracy1, out_class_loss1, out_malignancy_loss1 = [], [], [], []
    val_loss1, val_out_class_binary_accuracy1, val_out_class_loss1, val_out_malignancy_loss1 = [], [], [], []

    loss2, out_class_binary_accuracy2, out_class_loss2, out_malignancy_loss2 = [], [], [], []
    val_loss2, val_out_class_binary_accuracy2, val_out_class_loss2, val_out_malignancy_loss2 = [], [], [], []

    for index, row in archi1_result.iterrows():
        loss1.append(row["loss"])
        out_class_binary_accuracy1.append(row["out_class_binary_accuracy"])
        out_class_loss1.append(row["out_class_loss"])
        out_malignancy_loss1.append(row["out_malignancy_loss"])

        val_loss1.append(row["val_loss"])
        val_out_class_binary_accuracy1.append(row["val_out_class_binary_accuracy"])
        val_out_class_loss1.append(row["val_out_class_loss"])
        val_out_malignancy_loss1.append(row["val_out_malignancy_loss"])

    for index, row in archi2_result.iterrows():
        loss2.append(row["loss"])
        out_class_binary_accuracy2.append(row["out_class_binary_accuracy"])
        out_class_loss2.append(row["out_class_loss"])
        out_malignancy_loss2.append(row["out_malignancy_loss"])

        val_loss2.append(row["val_loss"])
        val_out_class_binary_accuracy2.append(row["val_out_class_binary_accuracy"])
        val_out_class_loss2.append(row["val_out_class_loss"])
        val_out_malignancy_loss2.append(row["val_out_malignancy_loss"])

    fig = plt.figure(figsize=(12, 7.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True)

    # archi1
    ax.plot(out_class_binary_accuracy1, color='orange', label='out_class_binary_accuracy', linewidth=1)
    ax.plot(loss1, color='blue', label='loss', linewidth=1)
    ax.plot(out_class_loss1, color='green', label='out_class_loss', linewidth=1)
    ax.plot(out_malignancy_loss1, color='red', label='out_malignancy_loss', linewidth=1)

    ax.plot(val_out_class_binary_accuracy1, color='orange', label='val_out_class_binary_accuracy', linewidth=1.6,
            linestyle='--')
    ax.plot(val_loss1, color='blue', label='val_loss', linewidth=1.6, linestyle='--')
    ax.plot(val_out_class_loss1, color='green', label='val_out_class_loss', linewidth=1.6, linestyle='--')
    ax.plot(val_out_malignancy_loss1, color='red', label='val_out_malignancy_loss', linewidth=1.6, linestyle='--')

    # archi2
    ax.plot(out_class_binary_accuracy2, color='black', label='out_class_binary_accuracy', linewidth=1)
    ax.plot(loss2, color='pink', label='loss', linewidth=1)
    ax.plot(out_class_loss2, color='purple', label='out_class_loss', linewidth=1)
    ax.plot(out_malignancy_loss2, color='grey', label='out_malignancy_loss', linewidth=1)

    ax.plot(val_out_class_binary_accuracy2, color='black', label='val_out_class_binary_accuracy', linewidth=1.6,
            linestyle='--')
    ax.plot(val_loss2, color='pink', label='val_loss', linewidth=1.6, linestyle='--')
    ax.plot(val_out_class_loss2, color='purple', label='val_out_class_loss', linewidth=1.6, linestyle='--')
    ax.plot(val_out_malignancy_loss2, color='grey', label='val_out_malignancy_loss', linewidth=1.6, linestyle='--')

    plt.title(model1_name + ' vs ' + model2_name)
    # title.get_title().set_fontsize(fontsize=20)
    plt.xlabel('epoch')
    plt.ylabel('accuracy  &  loss')
    plt.xlim(0, 11)
    plt.ylim(0)
    plt.xticks(np.linspace(0, 11, 12))
    plt.yticks(np.linspace(0, 3, 10))

    legend = plt.legend(['out_class_binary_accuracy1', 'loss1', 'out_class_loss1', 'out_malignancy_loss1',
                         'val_out_class_binary_accuracy1', 'val_loss1', 'val_out_class_loss1', 'val_out_malignancy_loss1',
                         'out_class_binary_accuracy2', 'loss2', 'out_class_loss2', 'out_malignancy_loss2',
                         'val_out_class_binary_accuracy2', 'val_loss2', 'val_out_class_loss2', 'val_out_malignancy_loss2'],
                        loc='upper right', ncol=2)
    legend.get_title().set_fontsize(fontsize=0.5)


def get_csv(directory, model_name=" ", train_full_set=True, manual_labels=True, ndsb3_holdout=0):
    holdout_txt = "_h" + str(ndsb3_holdout) if manual_labels else ""
    if train_full_set:
        holdout_txt = "_fs" + holdout_txt

    return directory + 'model_' + model_name + "_" + holdout_txt + "history.csv"


def plot_epoch_performance(directory='/opt/data/deeplearning/train_result/3DCNN/workdir/', network='3DCNN'):
    plot_result_diff_data(get_csv(directory, model_name="luna_posnegndsb_v1", manual_labels=True, ndsb3_holdout=0), model_name=network+'/luna_posnegndsb_v1_h0')
    plot_result_diff_data(get_csv(directory, model_name="luna_posnegndsb_v1", manual_labels=True, ndsb3_holdout=1), model_name=network+'/luna_posnegndsb_v1_h1')

    plot_result_diff_data(get_csv(directory, model_name="luna_posnegndsb_v2", manual_labels=True, ndsb3_holdout=0), model_name=network+'/luna_posnegndsb_v2_h0')
    plot_result_diff_data(get_csv(directory, model_name="luna_posnegndsb_v2", manual_labels=True, ndsb3_holdout=1), model_name=network+'/luna_posnegndsb_v2_h1')

    plot_result_diff_data(get_csv(directory, model_name="luna16_full", manual_labels=False, ndsb3_holdout=0), model_name=network+'/luna16_full')


def plot_network_performance(rootdirectory="/home/meditool/lung_cancer_detecor/src/"):
    # plot_result_diff_archi(
    #     get_csv(rootdirectory + '3DCNN/workdir/', model_name="luna_posnegndsb_v1", manual_labels=True, ndsb3_holdout=0),
    #     '3DCNN/luna_posnegndsb_v1_h0',
    #     get_csv(rootdirectory + 'ResNet50/workdir/', model_name="luna_posnegndsb_v1", manual_labels=True, ndsb3_holdout=0),
    #     'ResNet50/luna_posnegndsb_v1_h0')
    # plot_result_diff_archi(
    #     get_csv(rootdirectory + '3DCNN/workdir/', model_name="luna_posnegndsb_v1", manual_labels=True, ndsb3_holdout=1),
    #     '3DCNN/luna_posnegndsb_v1_h1',
    #     get_csv(rootdirectory + 'ResNet50/workdir/', model_name="luna_posnegndsb_v1", manual_labels=True, ndsb3_holdout=1),
    #     'ResNet50/luna_posnegndsb_v1_h1')
    #
    # plot_result_diff_archi(
    #     get_csv(rootdirectory + '3DCNN/workdir/', model_name="luna_posnegndsb_v2", manual_labels=True, ndsb3_holdout=0),
    #     '3DCNN/luna_posnegndsb_v2_h0',
    #     get_csv(rootdirectory + 'ResNet50/workdir/', model_name="luna_posnegndsb_v2", manual_labels=True, ndsb3_holdout=0),
    #     'ResNet50/luna_posnegndsb_v2_h0')
    # plot_result_diff_archi(
    #     get_csv(rootdirectory + '3DCNN/workdir/', model_name="luna_posnegndsb_v2", manual_labels=True, ndsb3_holdout=1),
    #     '3DCNN/luna_posnegndsb_v2_h1',
    #     get_csv(rootdirectory + 'ResNet50/workdir/', model_name="luna_posnegndsb_v2", manual_labels=True, ndsb3_holdout=1),
    #     'ResNet50/luna_posnegndsb_v2_h1')

    plot_result_diff_archi(
        get_csv(rootdirectory + 'workdir_1_5/', model_name="luna16_full", manual_labels=False, ndsb3_holdout=0),
        '3DCNN_1_5/luna16_full',
        get_csv(rootdirectory + 'workdir_1_10/', model_name="luna16_full", manual_labels=False, ndsb3_holdout=0),
        '3DCNN_1_10/luna16_full')


def choose_which_situation(diff_data_same_archi, same_data_diff_archi):
    if diff_data_same_archi:
        plot_epoch_performance()

    if same_data_diff_archi:
        plot_network_performance()


if __name__ == '__main__':

    # choose_which_situation(diff_data_same_archi=True, same_data_diff_archi=False)
    # choose_which_situation(diff_data_same_archi=True, same_data_diff_archi=False)
    # plot_epoch_performance()
    plot_network_performance()

    plt.show()
