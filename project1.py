# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 21:58:25 2022

@author: User
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from tensorflow.keras.optimizers import SGD
import keras
from keras.callbacks import EarlyStopping, TensorBoard
filepath =  r"C:\Users\User\Downloads\heart\heart.csv"

heart_disease = pd.read_csv(filepath)
heart_disease.head(10)
heart_disease = heart_disease.drop(['age','sex'], axis = 1)
heart_disease.head(10)
heart_disease_features = heart_disease.drop('target',axis=1)
heart_disease_label = heart_disease['target']
print(f"The shape of features: {heart_disease_features.shape}")
print(f"The shape of label: {heart_disease_label.shape}")

#5. One hot encode label
#convert to number encoding
heart_disease_label_OH = pd.get_dummies(heart_disease_label)

#Check the one-hot label
print("---------------One-hot Label-----------------")
print(heart_disease_label_OH.shape)

numpy_features = heart_disease_features.to_numpy()
numpy_label = heart_disease_label_OH.to_numpy()
print(f"The shape of features: {numpy_features.shape}")
print(f"The shape of label: {numpy_label.shape}")

SEED = 12345
features_train, features_iter, label_train, label_iter = train_test_split(numpy_features,
                                                                         numpy_label,
                                                                         test_size = 0.4,
                                                                         random_state = SEED)

features_val, features_test, label_val, label_test = train_test_split(features_iter,label_iter,
                                                                      test_size=0.5,
                                                                      random_state=SEED)
standardizer = StandardScaler()

features_train = standardizer.fit_transform(features_train)
features_val = standardizer.transform(features_val)
features_test = standardizer.transform(features_test)
print(f"Features train shape: {features_train.shape},Label train shape: {label_train.shape}")
print(f"Features validation shape: {features_val.shape}, Label validation shape: {label_val.shape}")
print(f"Features test shape: {features_test.shape}, Label test shape: {label_test.shape}")

adam = keras.optimizers.Adam(learning_rate=0.001)
bce = keras.losses.BinaryCrossentropy(from_logits=False)
accuracy = keras.metrics.BinaryAccuracy()
fnn_model = keras.Sequential([
    keras.layers.InputLayer(input_shape = features_train.shape[1]),
    keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.L2(0.001)),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.L2(0.001)),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='relu',kernel_regularizer=regularizers.L2(0.001)),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(label_train.shape[1], activation='sigmoid')
])
fnn_model.summary()

#model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])
fnn_model.compile(optimizer=adam,loss=bce,metrics=[accuracy])
base_log_path = r"C:\Program Files\models-master\research\tf_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
es = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
tb = TensorBoard(log_dir=log_path)

EPOCHS = 20
BATCH_SIZE = 32
history = fnn_model.fit(features_train,label_train,
                        validation_data=(features_val, label_val),
                        batch_size=BATCH_SIZE, epochs=EPOCHS, 
                        callbacks=[tb, es])

#Evaluate with test data for wild testing
test_result = fnn_model.evaluate(features_test,label_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")

#Make prediction
predictions_softmax = fnn_model.predict(features_test)
predictions = np.argmax(predictions_softmax,axis=-1)
label_test_element, label_test_idx = np.where(np.array(label_test) == 1)
for prediction, label in zip(predictions,label_test_idx):
    print(f'Prediction: {prediction} Label: {label}, Difference: {prediction-label}')
