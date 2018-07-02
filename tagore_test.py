
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD,RMSprop,adam
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.layers import Reshape
from keras.constraints import maxnorm





data_path_tagore = '/home/rupak/Technology/python/MachineLearning/keras/cnn/tagore_cnn/test'

data_dir_list = os.listdir(data_path_tagore)


#print("____________________________________________________________________")
#print(data_dir_list_n)

img_rows=128
img_cols=128
num_channel=1
num_epoch=20


img_data_list=[]





for dataset in data_dir_list:
        #print("************************"+str(data_path)+"******************"+str(dataset))
        img_list=os.listdir(data_path_tagore+'/')#dataset
        #print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        #print("*******************************"+str(img_list))
        for img in img_list:
                input_img=cv2.imread(data_path_tagore + '/'+ dataset)
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(128,128))
                img_data_list.append(input_img_resize)


tagore_len=len(img_data_list)# for array indexing
#print("length of tagore data--"+str(tagore_len))



img_data = np.array(img_data_list)
#print(img_data)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data[0])


names = ['<<<<<It is a picture of Rabindranath Tagore>>>>>','<<<<<it is not a picture Rabindranath>>>>>']
print(names)









input_shape=img_data.shape
print("###########################################")
print(input_shape)

# Define the number of classes
num_classes = 2






# Create the model
model = Sequential()
model.add(Reshape((1, 128, 128), input_shape=input_shape))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))





filename = "weights--21-0.0001.hdf5"#file name needs to be replaced
model.load_weights(filename)

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
print (img_data.shape)



X1= np.expand_dims(img_data, axis=0)
prediction=model.predict(X1, verbose=0)
print(prediction[0])

threshold=np.float32(0.5)

if(prediction[0][0]>threshold):
    print(names[1])
if(prediction[0][0]<threshold):
    print(names[0])
if(prediction[0][1]<threshold):
    print(names[1])
if(prediction[0][1]>threshold):
    print(names[0])














