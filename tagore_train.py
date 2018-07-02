
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



data_path_tagore = '/home/rupak/Technology/python/MachineLearning/keras/cnn/tagore_cnn/tagore'
data_path_not_tagore = '/home/rupak/Technology/python/MachineLearning/keras/cnn/tagore_cnn/not_tagore'
data_dir_list = os.listdir(data_path_tagore)
data_dir_list_n = os.listdir(data_path_not_tagore)

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

for dataset_n in data_dir_list_n:
        #print("************************"+str(data_path)+"******************"+str(dataset))
        img_list_n=os.listdir(data_path_not_tagore+'/')#dataset
        #print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        #print("*******************************"+str(img_list))
        for img in img_list_n:
                input_img=cv2.imread(data_path_not_tagore + '/'+ dataset_n)
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(128,128))
                img_data_list.append(input_img_resize)


n_tagore_len=len(img_data_list)-tagore_len
#print("length of not tagore data--"+str(n_tagore_len))


img_data = np.array(img_data_list)
#print(img_data)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)


num_of_samples = img_data.shape[0]


labels = np.zeros((num_of_samples,),dtype='int64')

print("  ***** labels length-- "+str(len(labels))+"   n_tagore_len-- *** "+str(n_tagore_len)+" tagore_len-- *** "+str(tagore_len))
labels[0:tagore_len]=1
labels[tagore_len:n_tagore_len-1]=0

#print(labels[5038])
#print(labels[5039])
#print(labels[5040])
#print(labels[5041])
#print(labels[5042])
#print(labels[5043])
#print(str(labels[5044])+"  "+str(labels[8176]))
#print(labels)
names = ['Rabindranath Tagore','Not Rabindranath']
print(names)


# Define the number of classes
num_classes = 2
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
print(Y)



#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
#print(X_train[0])







input_shape=img_data[0].shape
print(input_shape)



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
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())








# define the checkpoint
filepath="weights--{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=callbacks_list)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))






















