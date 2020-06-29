#loading the saved numpy arrays
import numpy as np
data=np.load('data.npy')
target=np.load('target.npy')

# Part 2 - Building the CNN
# Initialising the CNN
import tensorflow as tf
cnn = tf.keras.models.Sequential()#same as ann

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=200, kernel_size=3, activation='relu', input_shape=data.shape[1:]))
'''Adding the first layer-conv layer....filters-> no of feature detectors , 32 is classic , kernal size-> size of the feature detector matrix , here 3*3 is taken
input shape-> initialization of the first layer with the size of image , 3 is for colour images , 1 for b/w images , only used in the first layer
activation ->relu ,same as in ann'''

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
'''adding the maxpool layer
poolsize-> the size of the pooler matrix #check the intuition for more details
stride-> the no of pixels by which the pooler is shifted'''

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=100, kernel_size=3, activation='relu'))# we dont use the input_shape parameter in the intermediate layers
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

'''from here onwards same as ann'''
# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=64  , activation='relu'))# we chose more no of neurons as it is image

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))#since the output is binary based , we need only 1 neuron


# Part 3 - Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
#to save the best epoch model based on the val_loss...it is saved as .model file or as a folder in the repository
history=cnn.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

