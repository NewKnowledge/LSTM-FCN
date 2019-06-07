import pandas as pd
import os
import numpy as np
from all_datasets_training import generate_lstmfcn, generate_alstmfcn
from utils.keras_utils import train_model, evaluate_model
from utils.layer_utils import AttentionLSTM
from keras import backend as K
import pickle

EMBEDDINGS = True
if EMBEDDINGS:
    run_prefix = 'dry_run_keras_2_embeddings'
else: 
    run_prefix = 'dry_run_sin_2_embeddings'

# load saved training data

X_test = np.load("historical_dry_run_data/prepared/test_X_4_hrs.npy", allow_pickle = True)
y_test = np.load("historical_dry_run_data/prepared/test_y_4_hrs.npy", allow_pickle = True)

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

MAX_SEQUENCE_LENGTH = len(X_test[0][-1])
NB_CLASS = 2

# release GPU Memory
#K.clear_session()

# comment out the training code to only evaluate !
model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell, EMBEDDINGS)
#train_model(model, X_train, y_train, run_prefix, epochs=500, batch_size=128, val_split=1/4, embeddings=EMBEDDINGS)

f = evaluate_model(model, X_test, y_test, run_prefix, batch_size=128, embeddings=EMBEDDINGS)

file.write(f'test f1 historical dry run: {f}')
file.flush()
file.close()

