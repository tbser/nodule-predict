from keras.callbacks import Callback
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
import os
import numpy
import pandas

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):

        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)


class DNN_model(object):
    def __init__(self):
        self.MODEL_SUMMARY_FILE = './model_summary.txt'
        self.LEARN_RATE = 0.001
        self.TENSORBOARD_LOG_DIR = "tfb_log/"
        self.callbacks = []
        self.tensorboard_enabled = True
        self.logger = None
        self.model = None
        self.trained = False

    def writemodelsummary(self, s):
        with open(self.MODEL_SUMMARY_FILE, 'a') as f:
            f.write(s + '\n')
            print(s)

    def model_summary(self, model):
        open(self.MODEL_SUMMARY_FILE, "w")
        model.summary(print_fn=self.writemodelsummary)

    def step_decay(self, epoch):
        res = 0.001
        if epoch > 5:
            res = 0.0001
        if self.logger is not None:
            self.logger.info("learnrate: {0} epoch: {1}".format(res, epoch))
        return res

    def analysis_filename(self, file_name):
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

    def train(self, model_name, train_data_generator, trian_data_size, validate_data_generator,
              validate_data_size, epoch_number, train_epoch_save_folder, batch_size=16):
        if self.model is None:
            self.logger.error("The model is None. No training happens.")
            return

        if self.trained:
            return

        logcallback = LoggingCallback(self.logger.info)
        self.callbacks.append(logcallback)

        learnrate_scheduler = LearningRateScheduler(self.step_decay)
        self.callbacks.append(learnrate_scheduler)

        if self.tensorboard_enabled:
            if not os.path.exists(self.TENSORBOARD_LOG_DIR):
                os.makedirs(self.TENSORBOARD_LOG_DIR)

            tensorboard_callback = TensorBoard(
                log_dir=self.TENSORBOARD_LOG_DIR,
                histogram_freq=2,
                # write_images=True, # Enabling this line would require more than 5 GB at each `histogram_freq` epoch.
                write_graph=True
                # embeddings_freq=3,
                # embeddings_layer_names=list(embeddings_metadata.keys()),
                # embeddings_metadata=embeddings_metadata
            )
            tensorboard_callback.set_model(self.model)
            self.callbacks.append(tensorboard_callback)

        checkpoint_epoch_model = ModelCheckpoint(
            train_epoch_save_folder + "model_" + model_name + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
            monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
            period=1)
        checkpoint_best_model = ModelCheckpoint(
            train_epoch_save_folder + "model_" + model_name + "_best.hd5", monitor='val_loss',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.callbacks.append(checkpoint_epoch_model)
        self.callbacks.append(checkpoint_best_model)

        train_history = self.model.fit_generator(train_data_generator, trian_data_size / batch_size, epoch_number,
                                                 validation_data=validate_data_generator,
                                                 validation_steps=validate_data_size / batch_size,
                                                 callbacks=self.callbacks)
        self.logger.info("Model fit_generator finished.")
        self.model.save(train_epoch_save_folder + "model_" + model_name + "_end.hd5")

        pandas.DataFrame(train_history.history).to_csv(
            train_epoch_save_folder + "model_" + model_name + "_history.csv")
        self.trained = True


    def predict(self, img_list):
        if self.model is None:
            self.logger.error("The model is None. Please call generate_model() to generate the model at first.")
            return
        batch_size = 1  # for test
        batch_list = []
        batch_list_loc = []
        count = 0
        predictions = []

        for item in img_list:
            cube_img = item[0]
            file_name = item[1]
            patient_id = self.analysis_filename(file_name)[0]
            self.logger.info("====={0} - patient_id {1}".format(count, patient_id))
            # logger.info("the shape of cube image: {0}".format(numpy.array(cube_img).shape)) # (1, 32, 32, 32, 1)
            count += 1
            batch_list.append(cube_img)
            batch_list_loc.append(file_name)
            # logger.info("batch list: {0}".format(batch_list))
            # logger.info("the shape of batch list: {0}".format(numpy.array(batch_list).shape)) # (1, 1, 32, 32, 32, 1)
            # logger.info("batch list loc: {0}".format(batch_list_loc))

            # if len(batch_list) % batch_size == 0:
            batch_data = numpy.vstack(batch_list)
            p = self.model.predict(batch_data, batch_size=batch_size)
            self.logger.info("the prediction result p: {0}".format(p))
            # [array([[ 0.00064842]], dtype=float32), array([[  1.68593288e-05]], dtype=float32)]
            self.logger.info("the shape of p:{0}".format(numpy.array(p).shape))  # (2, 1, 1)
            self.logger.info("the length of p[0]:{0}".format(len(p[0])))  # 1

            # for i in range(len(p[0])):
            i = 0
            file_name = batch_list_loc[i]
            nodule_chance = p[0][i][0]
            diameter_mm = round(p[1][i][0], 4)
            nodule_chance = round(nodule_chance, 4)
            self.logger.info("nodule chance:{0}, diameter_mm:{1}".format(nodule_chance, diameter_mm))
            item_prediction = [file_name, nodule_chance, diameter_mm]
            predictions.append(item_prediction)

            batch_list = []
            batch_list_loc = []
            # count = 0

        return predictions
