# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:28:00 2018
Kaggle Dog Breed Identification using Keras running Tensorflow backend
Doggies are cute, Woof Woof Woof 

@author: Carmen Su 
"""
import os
import cv2 # image handling
import pandas as pd
import numpy as np 
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation, Flatten

print ('Examine the labels & set Y_train')
lables = pd.read_csv('labels.csv')
print (lables.head(5))
breed_count = lables['breed'].value_counts()
print (breed_count.head())
print (breed_count.shape)

# breed visualisation 
ax = pd.value_counts(lables['breed'],ascending=True).plot(kind='barh',
                                                       fontsize="40",
                                                       title="Class Distribution",
                                                       figsize=(50,100))
ax.set(xlabel="Images per class", ylabel="Classes")
ax.xaxis.label.set_size(40)
ax.yaxis.label.set_size(40)
ax.title.set_size(60)

# target one hot encoding 
targets = pd.Series(lables['breed'])
one_hot = pd.get_dummies(targets, sparse = True)
one_hot_labels = np.asarray(one_hot)
        
print ('Import trainning images & and tranfer them into arrays')
img_rows = 128
img_cols = 128
num_channel = 1 # 3 colour channes

# testing cv2 for a single image
path = 'C:\\Users\\jiawe\\Dropbox\\Data Science Projects\\Dog'
'''
img_1 = cv2.imread(path+'\\train\\'+'000bec180eb18c7604dcecc8fe0dba07.jpg', 0)
img_1_resize= cv2.resize(img_1, (img_rows, img_cols)) 
cv2.imshow('Single Image', img_1_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

print ('image listing, iterate all images in the train folder')
x_feature = []
y_feature = []

i = 0 # initialisation
for f, img in tqdm(lables.values): # f for format ,jpg
    train_img = cv2.imread(path + '\\train\\{}.jpg'.format(f),0)
    label = one_hot_labels[i]
    train_img_resize = cv2.resize(train_img, (img_rows, img_cols)) 
    x_feature.append(train_img_resize)
    y_feature.append(label)
    i += 1
    
print ('transform data each datum into an array')
x_train_data = np.array(x_feature, np.float32) / 255.   # /= 255 for normolisation
x_train_data = np.expand_dims(x_train_data, axis = 3)
y_train_data = np.array(y_feature, np.uint8)
print (x_train_data.shape)
print (y_train_data.shape)

num_class = y_train_data.shape[1]

x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_data, test_size=0.1,  random_state=2)

print ('prepare test data')        
submission = pd.read_csv('sample_submission.csv')
test_img = submission['id']

x_test_feature = []

i = 0 # initialisation
for f in tqdm(test_img.values): # f for format ,jpg
    img = cv2.imread(path + '\\test\\{}.jpg'.format(f),0)
    img_resize = cv2.resize(img, (img_rows, img_cols)) 
    x_test_feature.append(img_resize)  
    
x_test_data = np.array(x_test_feature, np.float32) / 255. 
x_test_data = np.expand_dims(x_test_data, axis = 3)
print (x_test_data.shape)
print ('Building the model')
model = Sequential()

# add convolution layer
# retifier ensure the non-linearity in the processing 
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same', 
                         activation ='relu', input_shape = (img_rows, img_cols, num_channel))) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same', 
                         activation ='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) 
# ANN 
model.add(Dense(units = 120, activation = 'relu')) 
# output layer
model.add(Dense(units = 120, activation = 'softmax'))   
        
# compiling the model
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])     
model.summary()

# feed images into CNN
batch_size = 256 
nb_epochs = 10
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    verbose=2, 
                    validation_data=(x_val, y_val),
                    initial_epoch=0)

print ('Plot the loss and accuracy curves for training and validation ')
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# predict results
results = model.predict(x_test_data)
prediction = pd.DataFrame(results)

# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
prediction.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
prediction.insert(0, 'id', submission['id'])

submission = prediction
submission.to_csv('new_submission.csv', index=False)
