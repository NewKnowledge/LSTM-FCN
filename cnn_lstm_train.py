import pandas as pd
import os
import numpy as np
from all_datasets_training import generate_lstmfcn, generate_alstmfcn
from utils.keras_utils import train_model, evaluate_model
from utils.layer_utils import AttentionLSTM
from keras import backend as K
import pickle

run_prefix = 'dry_run_initial_model'

# load saved training data
X_train = np.load("dry_run_data/prepared/train_X.npy", allow_pickle = True)
y_train = np.load("dry_run_data/prepared/train_y.npy", allow_pickle = True)
X_test = np.load("dry_run_data/prepared/test_X.npy", allow_pickle = True)
y_test = np.load("dry_run_data/prepared/test_y.npy", allow_pickle = True)

# use just rate values as features
X_train = np.vstack([feature_vector[-1] for feature_vector in X_train])
X_test = np.vstack([feature_vector[-1] for feature_vector in X_test])
X_train = X_train.reshape(-1,1,X_train.shape[1])
X_test = X_test.reshape(-1,1,X_test.shape[1])

# model parameters
MODEL_NAME = 'alstmfcn'
model_fn = generate_alstmfcn
cell = 128

# results directories
base_log_name = '%s_%d_cells.csv'
base_weights_dir = '%s_%d_cells_weights/'
if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
    file = open(base_log_name % (MODEL_NAME, cell), 'w')
    file.write('%s\n' % ('test_accuracy'))
    file.close()
file = open(base_log_name % (MODEL_NAME, cell), 'a+')

MAX_SEQUENCE_LENGTH = X_train.shape[2]
NB_CLASS = len(np.unique(y_train))

# release GPU Memory
#K.clear_session()

# comment out the training code to only evaluate !
model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)
#train_model(model, X_train, y_train, run_prefix, epochs=500, batch_size=128)

acc = evaluate_model(model, X_test, y_test, run_prefix, batch_size=128)

file.write("%0.6f\n" % (acc))
file.flush()
file.close()
