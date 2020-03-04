'''
Machine Learning with Ultrasound Data

Written by Sam Epeagba
02/12/2020

References:
    Keras Data generator: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Data Orginization: https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
'''
#import matplotlib.pyplot as plt

from keras.models import load_model
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
#from my_classes import DataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import cv2

dataPath = '/home/sepeagb2/sam/data/Ultrasound-Machine-Learning-Research/test.csv'
image_path='/home/sepeagb2/sam/data/Ultrasound-Machine-Learning-Research/combined/' 

def dataSegment(path):
    data = pd.read_csv(dataPath)
    X = [i for i in range(len([j for j in data['id']]))]
    Y = X
    X_train, X_val_holder, Y_train, Y_val_holder = train_test_split(X, Y,test_size=0.4, random_state=None)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_holder, Y_val_holder, test_size=0.5, random_state=None)
    partition1 = {'X_train':X_train, 'X_val':X_val, 'X_test': X_test}
    return partition1, data

def loadDataList():
    partition, data = dataSegment(dataPath)
    X_id_val = partition['X_train']
    random.shuffle(X_id_val)
    Y_id_val = partition['X_train']
    Validation_id_val_x=partition['X_val']
    Validation_id_val_y=partition['X_val']
    Test_id_val_x=partition['X_test']
    Test_id_val_y=partition['X_test']
    #X_image = [data['id'][i] for i in X_id_val]
    #random.shuffle(X_image)
    #Y_label = [data['label'][i] for i in Y_id_val]
    #return X_image, Y_label
    return X_id_val, Y_id_val, Validation_id_val_x, Validation_id_val_y, Test_id_val_x, Test_id_val_y, data

def prepareData():
    x,y,val_x,val_y,test_x,test_y,data = loadDataList()
    #image_path='/home/sepeagb2/sam/data/Ultrasound-Machine-Learning-Research/combined/' 
    X=[]
    Y=[]
    Val_x=[]
    Val_y=[]
    Test_x=[]
    Test_y=[]

    for image_id in x:
        X.append(cv2.resize(cv2.imread(image_path+data['id'][image_id], cv2.IMREAD_COLOR),(320,240), interpolation=cv2.INTER_CUBIC))

        #X.append(cv2.imread(image_path+data['id'][image_id]))
        Y.append(data['label'][image_id])
    for image_id in val_x:
        Val_x.append(cv2.resize(cv2.imread(image_path+data['id'][image_id], cv2.IMREAD_COLOR),(320,240), interpolation=cv2.INTER_CUBIC))

        #Val_x.append(cv2.imread(image_path+data['id'][image_id]))
        Val_y.append(data['label'][image_id])
    for image_id in test_x:
        Test_x.append(cv2.resize(cv2.imread(image_path+data['id'][image_id], cv2.IMREAD_COLOR),(320,240), interpolation=cv2.INTER_CUBIC))

        #Test_x.append(cv2.imread(image_path+data['id'][image_id]))
        Test_y.append(data['label'][image_id])
    return X, Y, Val_x, Val_y, Test_x, Test_y

def convertArry():
    X,Y,Val_x,Val_y, Test_x,Test_y = prepareData()
    X_num = np.array(X)
    Y_num = np.array(Y)
    #Y_num=Y
    Val_x_num=np.array(Val_x)
    Val_y_num=np.array(Val_y)
    Test_x_num=np.array(Test_x)
    Test_y_num=np.array(Test_y)
    
    Y_mat=to_categorical(Y_num)
    Val_y_mat=to_categorical(Val_y_num)
    Test_y_mat=to_categorical(Test_y_num)

    print "the shape of the X_input array is: " + str(X_num.shape)
    #print "the shape of Y_output array is: " + str(Y_mat.shape)
    print "the shape of Val_x array is: " + str(Val_x_num.shape)
    #print "the shape of Val_y array is: " + str(Val_y_mat.shape)
    print "the shape of Test_x array is: "+ str(Test_x_num.shape)
    #print "the hsape of test_y array is: "+ str(Test_y_mat.shape)
    print("------------------------------------------------------------------------") 
    print("------------------------------------------------------------------------") 
    print("------------------------------------------------------------------------") 
    print("------------------------------------------------------------------------")      
    
    print "the shape of Y_num array is: " + str(Y_num.shape)
    print "the shape of Val_y array is: " + str(Val_y_num.shape)
    print "the shape of Test_y array is: " + str(Test_y_num.shape)
    
    #return X_num, Y_mat, Val_x_num, Val_y_mat, Test_x_num, Test_y_mat
    return X_num, Y_num, Val_x_num, Val_y_num, Test_x_num, Test_y_num    

def cnnModel():
    model=models.Sequential()
    #model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(960,1280,3)))
    #model.add(layers.Conv2D(32,(3,3,), activation='relu', input_shape=(240,320,3)))
    #test to see if can add a 3,3,3 filter and use that as the thing
    model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(240,320,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))
    #model.add(layers.Dense(31, activation='softmax'))
    return model

def trainModel():
    x_train,y_train,val_x,val_y,test_x,test_y = convertArry()
    ntrain = len(y_train)
    nval = len(val_y)
    batch_size=30
    cnn = cnnModel()
    cnn.summary()
    #cnn.compile(loss = 'binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    cnn.compile(loss = 'mean_squared_error',optimizer=optimizers.SGD(lr=1e-4),metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0, width_shift_range=0,height_shift_range=0, shear_range=0, 
                                       zoom_range=0, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)
    val_generator=val_datagen.flow(val_x,val_y, batch_size=batch_size)

    history=cnn.fit_generator(train_generator, steps_per_epoch=ntrain//batch_size, epochs=100, validation_data=val_generator,
                            validation_steps=nval//batch_size, use_multiprocessing=True)
    #new version
    #history=cnn.fit(x_train,y_train,steps_per_epoch=ntrain/batch_size, epochs=64, validation_data=val_generator, validation_steps=nval/batch_size)
    model_score=cnn.evaluate(test_x,test_y)
    print model_score
    model_predict=cnn.predict(test_x)
    print model_predict
    print test_y
    #cnn.save_weights('/home/sepeagb2/sam/code/saved_weights/model_weights_1_bad.h5')
    #cnn.save('/home/sepeagb2/sam/code/saved_models/model_keras_1_bad.h5')

def main():
    trainModel()
    

if __name__=="__main__":    
    main()
    
