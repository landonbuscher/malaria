import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

im_path = 'C:\\Users\\lando\\Documents\\GitHub\\malaria\\cell_images'
test_path = im_path+'\\test'
train_path = im_path+'\\train'

dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'\\uninfected'):
    img = imread(test_path+'\\uninfected\\'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

image_shape = (int(np.round(np.mean(dim1))),int(np.round(np.mean(dim2))),3)

image_gen = ImageDataGenerator(rotation_range=20,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rescale=1/255,
                                shear_range=0.1,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                fill_mode='nearest')

model = Sequential([
    Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,
            activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,
            activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,
            activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',
                metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',patience=2)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                shuffle=False)

results = model.fit(train_image_gen,epochs=20,
                                validation_data=test_image_gen,
                                callbacks=[early_stop])

model.save('malaria_detector.h5')
