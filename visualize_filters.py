from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, CuDNNLSTM, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES
from utils.generic_utils import load_dataset_at
from utils.keras_utils import visualize_filters
from utils.layer_utils import AttentionLSTM

import os
import traceback
import json
from keras import backend as K
import numpy as np


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_attention_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == '__main__':
    # run prefix
    run_prefix = 'alstm_4_hr'

    # load rate functions
    series_size = 240 * 60
    num_bins = 300
    min_points = 5
    filter_bandwidth = 2
    density = True
    dir_path = "kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
    series_values =  np.load("../manatee/manatee/rate_values/" + dir_path + "/series_values.npy")
    # change this line from 'labels.npy' to 'labels_multi.npy' for binary vs. multiclass
    labels =  np.load("../manatee/manatee/rate_values/" + dir_path + "/labels.npy")

    # randomly shuffle before splitting into training / test / val
    np.random.seed(0)
    randomize = np.arange(len(series_values))
    np.random.shuffle(randomize)
    series_values = series_values[randomize]
    series_values = series_values.reshape(-1,1,series_values.shape[1])
    labels = labels[randomize]
    train_split = int(0.9 * series_values.shape[0])
    X_train, y_train  = series_values[:train_split], labels[:train_split]
    X_test, y_test = series_values[train_split:], labels[train_split:]

    # COMMON PARAMETERS
    #DATASET_ID = 0
    num_cells = 128
    model = generate_attention_lstmfcn  # Select model to build
    model_name = 'alstmfcn'

    # Visualizaion params
    CONV_ID = 0
    FILTER_ID = 100

    """ <<<<< SCRIPT SETUP >>>>> """
    # Script setup
    sequence_length = series_values.shape[-1]
    nb_classes = len(np.unique(labels))
    model = model(sequence_length, nb_classes, num_cells)
    visualize_filters(model, X_train, y_train, run_prefix, conv_id=CONV_ID, filter_id=FILTER_ID, seed=0)
