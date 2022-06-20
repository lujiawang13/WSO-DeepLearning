import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from WSODL_f import WSODL_f
from WSODL_f import WSODLbinary_f
from predict_f import predict_f

# Training data
Datapath='C:\\'
TrainData = pd.read_csv(Datapath+'TrainData.csv')
features=["sex", "length", 'diameter', 'height', 'whole_weight',
          'shucked_wieght', 'viscera_wieght', 'shell_weight' ]

# WSO Deep learning model
model = Sequential()
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(WSODLbinary_f())
#model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

model_info=WSODL_f(model,TrainData,features,epochs=5)

# make probability predictions with the model
TestData = pd.read_csv(Datapath+'TestData.csv')
labels=predict_f(model_info,TestData)
yTest = TestData.loc[:, 'y']
print(np.mean(abs(labels-np.array(yTest))))
