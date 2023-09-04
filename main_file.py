#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv', usecols=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

le = LabelEncoder()
train['answer'] = le.fit_transform(train['answer'])
def df_to_dataset(dataframe, shuffle=True, batch_size=16):
    df = dataframe.copy()
    labels = df.pop('answer')
    df = df['A'] #This is the all info the model will be training on. Make sure you select all narrative columns.
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size) #Grouping, can't be greater than the row number. Set it lower.
    ds = ds.prefetch(tf.data.AUTOTUNE)  
    return ds

train_set, val_set, test_set = np.split(train.sample(frac=1), [int(0.8*len(train)), int(0.9*len(train))])

train_data = df_to_dataset(train_set)
val_data = df_to_dataset(val_set)
test_data = df_to_dataset(test_set)

embedding_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = "https://tfhub.dev/google/nnlm-en-dim50/2"

hub_layer = hub.KerasLayer(embed, dtype=tf.string, trainable=True)


#Model and Tuning Part
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_data, epochs=5, validation_data=val_data)
