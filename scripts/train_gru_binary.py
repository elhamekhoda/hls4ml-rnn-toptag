
import pandas as pd
import numpy as np
import time
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, GRU,TimeDistributed, Conv1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
import ast
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from scipy.stats import uniform, truncnorm, randint
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from scipy.stats import loguniform
# from pandas import read_csv
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from datetime import datetime
today = datetime.date(datetime.now())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


outdir = "/global/homes/e/elham/rnn_hls4ml/results/"
outpath = outdir+"gru_training_%s/"%today
comd = "mkdir -p "+outpath
os.system(comd)

# inputdir = 'rnn_hls4ml_data_scaled'
inputdir = 'rnn_hls4ml_data_scaled_2class'


'''
all_features = np.load('/global/cscratch1/sd/elham/'+inputdir+'/jet-tag_features_2class.npy')
all_labels = np.load('/global/cscratch1/sd/elham/'+inputdir+'/jet-tag_labels_2class.npy')


x_train, x_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.05, random_state=42)

np.save('/global/cscratch1/sd/elham/'+inputdir+'/x_test.npy', x_test)
np.save('/global/cscratch1/sd/elham/'+inputdir+'/y_test.npy', y_test)
np.save('/global/cscratch1/sd/elham/'+inputdir+'/x_train.npy', x_train)
np.save('/global/cscratch1/sd/elham/'+inputdir+'/y_train.npy', y_train)
'''

x_train = np.load('/global/cscratch1/sd/elham/'+inputdir+'/x_train.npy', 'r')
y_train = np.load('/global/cscratch1/sd/elham/'+inputdir+'/y_train.npy', 'r')

# x_train = x_train[:,:15,:]
y_train = y_train[:,4:5]
plt.hist(y_train)
plt.savefig('%s/y_train.png'%(outpath))

print("y-shape: ", y_train.shape)
print("x-shape: ", x_train.shape)


################################################################################
#                        Small model
#
################################################################################

model = Sequential()
model.add(GRU(5, kernel_initializer = 'VarianceScaling', kernel_regularizer = regularizers.l1_l2(l1= 0.00001, l2 = 0.0001), name = 'layer1', input_shape = (20,6)))
# model.add(Flatten(name='layer2'))
model.add(Dense(5, activation='relu', kernel_initializer='glorot_normal', name='layer2'))
model.add(Dense(1, activation='sigmoid', name = 'output_sigmoid'))


model.summary()

es = EarlyStopping(monitor='val_loss',min_delta = 1e-4, mode='min', verbose=1, patience=10)
adam = Adam(lr = 0.0002)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(x_train.astype('float32'), y_train.astype('float32'), 
                    batch_size = 256,
                    epochs = 150, 
                    validation_split = 0.2, 
                    shuffle = True,
                    callbacks = [ModelCheckpoint(outpath+'model_toptag_gru.h5', verbose=1, save_best_only=True), es],
                    use_multiprocessing=True, workers=4)

np.savetxt(outpath+'training_gru.txt', history.history['loss'])
np.savetxt(outpath+'validation_gru.txt', history.history['val_loss'])






