import pandas as pd
import os
import numpy as np
from all_datasets_training import generate_lstmfcn, generate_alstmfcn
from utils.keras_utils import train_model, evaluate_model
from utils.layer_utils import AttentionLSTM
from keras import backend as K
import pickle

run_prefix = 'alstm_4_hr'

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

# train
train_split = int(0.9 * series_values.shape[0])
X_train, y_train  = series_values[:train_split], labels[:train_split]
X_test, y_test = series_values[train_split:], labels[train_split:]

# test LSTM-FCN and attention models
MODELS = [
    #('lstmfcn', generate_lstmfcn),
    ('alstmfcn', generate_alstmfcn),
]

base_log_name = '%s_%d_cells.csv'
base_weights_dir = '%s_%d_cells_weights/'

# Number of cells
CELLS = [128]#[8, 64, 128]

# Normalization scheme
# Normalize = False means no normalization will be done
# Normalize = True / 1 means sample wise z-normalization
# Normalize = 2 means dataset normalization.
normalize_dataset = True

for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
    for cell in CELLS:
        successes = []
        failures = []

        if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
            file = open(base_log_name % (MODEL_NAME, cell), 'w')
            file.write('%s\n' % ('test_accuracy'))
            file.close()

        MAX_SEQUENCE_LENGTH = series_values.shape[-1]
        NB_CLASS = len(np.unique(labels))

        # release GPU Memory
        K.clear_session()

        file = open(base_log_name % (MODEL_NAME, cell), 'a+')

        weights_dir = base_weights_dir % (MODEL_NAME, cell)

        if not os.path.exists('../lstm_weights/' + weights_dir):
            os.makedirs('../lstm_weights/' + weights_dir)

        # try:
        model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

        # comment out the training code to only evaluate !
        #train_model(model, X_train, y_train, run_prefix, epochs=500, batch_size=128,
        #            normalize_timeseries=normalize_dataset)

        acc = evaluate_model(model, X_test, y_test, run_prefix, batch_size=128,
                                normalize_timeseries=normalize_dataset)

        s = "%0.6f\n" % (acc)

        file.write(s)
        file.flush()

        successes.append(s)

        # except Exception as e:
        #     traceback.print_exc()
        #
        #     s = "%d,%s,%s,%s\n" % (did, dname, dataset_name_, 0.0)
        #     failures.append(s)
        #
        #     print()

        file.close()

    print('\n\n')
    print('*' * 20, "Successes", '*' * 20)
    print()

    for line in successes:
        print(line)

    print('\n\n')
    print('*' * 20, "Failures", '*' * 20)
    print()

    for line in failures:
        print(line)