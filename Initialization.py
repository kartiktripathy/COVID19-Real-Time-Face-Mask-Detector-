#Initializing the data path and assigning the classes
import cv2 ,os
data_path = 'dataset'
classes = os.listdir(data_path)
labels = [i for i in range(len(classes))]
label_dict = dict(zip(classes,labels))

print(label_dict)
print(classes)
print(labels)

data=[]
target=[]


for classes in classes:
    folder_path=os.path.join(data_path,classes)
    img_names=os.listdir(folder_path)
        
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            #Coverting the image into gray scale
            resized=cv2.resize(gray,(100,100))
            #resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[classes])
            #appending the image and the label(categorized) into the list (dataset)

        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image

import numpy as np

data=np.array(data)/255.0 #Rescaling the images
data=np.reshape(data,(data.shape[0],100,100,1))
target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)

np.save('data',data)
np.save('target',new_target)
#We save the data and target as numpy arrays 
#these are uploaded in the zip file named as k.zip
