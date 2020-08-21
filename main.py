import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
model = load_model('C:\\Users\\lando\\Documents\\GitHub\\malaria\\malaria_detector.h5')
img = input('Enter path to cell image. Keep this image around 130px by 130px.\n')
img = image.load_img(img,target_size=(130,130,3))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
pred = int(np.round(model.predict(img)[0][0]))
if pred == 1:
    print('Cell is uninfected.')
else:
    print('Cell is infected.')
