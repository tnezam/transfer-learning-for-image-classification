import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
# import os
# os.add_dll_directory("C:\ Users\Taraneh\AppData\Local\Programs\Python\Python310")
import tensorflow as tf
from tensorflow import keras
# from keras import layers
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import cv2
import pathlib

dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = keras.utils.get_file('flower_photos', origin = dataset_url, untar=True) #Downloads 'flower_photos'(file/folder)from dataset_url and stores in data_dir
data_dir = pathlib.Path(data_dir) #Path class allows to create path objects(now methods can be used on it) which encodes information about location of files & directories in

roses = list(data_dir.glob('roses/*'))#list of files starts with'roses' in data_dir(glob is used to collect files in directory)
print(roses[0])#/root/.keras/datasets/flower_photos/roses/6409000675_6eb6806e59.jpg
PIL.image.open(str(roses[0]))

image_height,image_width = 180,180
batch_size = 32
train_ds = keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2, subset = 'training', seed =123,
                                                            label_mode = 'categorical', image_size = (image_height, image_width),
                                                            batch_size = batch_size)

val_ds = keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2, subset = 'validation', seed =123,
                                                            label_mode = 'categorical', image_size = (image_height, image_width),
                                                            batch_size = batch_size)


class_names = train_ds.class_names
print(class_names)

resnet_model = Sequential() #build a sequential model
pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(180,180,3),pooling='avg',classes=5,
                                               weights='resnet') #False means input & output layers is different from resnet
for layer in pretrained_model.layers:
    layer.trainable=False #layers in pretrained are fixed

resnet_model.add(pretrained_model)
resnet_model.add(Flatten()) # to vectorize
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))
resnet_model.summary()

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

epochs=10
resnet_model.fit(train_ds,validation_data=val_ds,epochs=epochs)


fig1=plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train','Validation'])
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Validation'])
plt.show()

import cv2
image=cv2.imread(str(roses[0]))
image_resized=cv2.resize(image,(image_height,image_width))
image=np.expand_dims(image_resized,axis=0)
print(image.shape)
prdiction=resnet_model.predict(image)
print(prdiction)
output_class=class_names[np.argmax(prdiction)]
print(f"The output Class is:{output_class}")