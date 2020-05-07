'''
Machine Learning with Ultrasound Data

Written by Sam Epeagba
02/12/2020

References:
    Keras Data generator: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Data Orginization: https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
    https://www.pyimagesearch.com/2017/10/30/how-to-multi-gpu-training-with-keras-python-and-deep-learning/
'''
#import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("Agg")
#from pyimagesearch.minigooglenet import MiniGoogLeNet
#from sklearn.preprocessing import LabelBinarizer
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import LearningRateScheduler
#from keras.utils.training_utils import multi_gpu_model

#import matplotlib.pyplot as plt
#import tensorflow as tf
#import numpy as np
#import argparse


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
from keras.applications.resnet50 import ResNet50
from keras.models import Model

print("Program starting")

dataPath = '/home/sepeagb2/sam/code/abstractResults/03152020/dataFinal.csv'


def dataSegment(dataPath):
    print("datasegment function accessed")
    data = pd.read_csv(dataPath)
    X = [i for i in range(len([j for j in data['id']]))]
    Y = X
    #X_train, X_val_holder, Y_train, Y_val_holder = train_test_split(X, Y,test_size=0.4, random_state=None)
    #X_val, X_test, Y_val, Y_test = train_test_split(X_val_holder, Y_val_holder, test_size=0.5, random_state=None)
    #partition1 = {'X_train':X_train, 'X_val':X_val, 'X_test': X_test}
    #print("returning value from dataSegmnet")
     
    X_train, X_val_holder, Y_train, Y_val_holder = train_test_split(X, Y,test_size=0.6, random_state=None)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_holder, Y_val_holder, test_size=0.5, random_state=None)
    partition1 = {'X_train':X_train, 'X_val':X_val, 'X_test': X_test}
    print("returning value from dataSegmnet")
    return partition1, data

def loadDataList():
    print("loadDataList function accessed")
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
    print("returning value from loadDataList")
    return X_id_val, Y_id_val, Validation_id_val_x, Validation_id_val_y, Test_id_val_x, Test_id_val_y, data

def prepareData():
    print("prepareData function accessed")
    x,y,val_x,val_y,test_x,test_y,data = loadDataList()
    print("the number X_train values")
    print x
    print("prepareData received value from loadDataList()")
    image_path='/home/sepeagb2/sam/code/abstractResults/03152020/combined4/' 
    X=[]
    Y=[]
    Val_x=[]
    Val_y=[]
    Test_x=[]
    Test_y=[]
    print("puting data into a list")
    val_add=0
    for image_id in x:
        val_add+=1
        print("Append to list: ",val_add,':',str(data['id'][image_id]))
 
        X.append(cv2.resize(cv2.imread(image_path+data['id'][image_id], cv2.IMREAD_COLOR),(320,240), interpolation=cv2.INTER_CUBIC))
        #X.append(cv2.imread(image_path+data['id'][image_id]))
        Y.append(data['label'][image_id])
        
        #print("Append to list: ",val_add,':',str(data['id'][image_id]))

    print("train_x and train_y completed and put into a list")
    val_add=0
    for image_id in val_x:
        val_add+=1
        print("Append to list: ",val_add,':',str(data['id'][image_id]))
        Val_x.append(cv2.resize(cv2.imread(image_path+data['id'][image_id], cv2.IMREAD_COLOR),(320,240), interpolation=cv2.INTER_CUBIC))

        #Val_x.append(cv2.imread(image_path+data['id'][image_id]))
        Val_y.append(data['label'][image_id])
        
        

    print("Val_x and Val_y completed and put into a list")
    val_add=0
    for image_id in test_x:
        val_add+=1
        print("Append to list: ",val_add,':',str(data['id'][image_id]))  
        Test_x.append(cv2.resize(cv2.imread(image_path+data['id'][image_id], cv2.IMREAD_COLOR),(320,240), interpolation=cv2.INTER_CUBIC))

        #Test_x.append(cv2.imread(image_path+data['id'][image_id]))
        Test_y.append(data['label'][image_id])
        
        

    print("test_y and test_y  completed and put into a list")
    print("returning value from prepareData")
    return X, Y, Val_x, Val_y, Test_x, Test_y

def convertArry():
    print("convertArry function accessed")
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
    #1d arry to 2darry
    Y_num=np.reshape(Y_num,(Y_num.size,1))
    Val_y_num=np.reshape(Val_y_num,(Val_y_num.size,1))
    Test_y_num=np.reshape(Test_y_num,(Test_y_num.size,1))

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
    print("return value from convertArry")
    return X_num, Y_num, Val_x_num, Val_y_num, Test_x_num, Test_y_num    

def cnnModel():
    print("accessed cnnModel()")
    IMG_HEIGHT=240
    IMG_WIDTH=320
    print("loading ResNet50")
    restnet=ResNet50(include_top=False, weights='imagenet',input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
    
    output=restnet.output
   # restnet=Model(restnet.input, output=output)

    restnet=Model(restnet.input, output=output)
    
    for layer in restnet.layers[15:]:
        layer.trainable=True
    restnet.summary()
    print("creating Sequential model()")
    model=models.Sequential()
    model.add(restnet)
    #adding vgg_net
    #model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(960,1280,3)))
    #model.add(layers.Conv2D(32,(3,3,), activation='relu', input_shape=(240,320,3)))
    #test to see if can add a 3,3,3 filter and use that as the thing
    #model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(240,320,3)))
    #model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Conv2D(64, (3,3),activation='relu'))
    #model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Conv2D(128,(3,3),activation='relu'))
    #model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='tanh',input_dim=3))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='tanh'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='linear'))
    #model.add(layers.Dense(31, activation='softmax'))
    print("returning model")
    return model

def cnnModelMulti():
    model=models.Sequential()
    #adding vgg_net
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
    model.add(layers.Dense(1, activation='tanh'))
    #model.add(layers.Dense(31, activation='softmax'))
    return model

def trainModel():
    print("trainModel function accessed")
    x_train,y_train,val_x,val_y,test_x,test_y = convertArry()
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")   
    print("------------------------------------------------------------------------")
    print(np.multiply(y_train,1./30))
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print(y_train/30) 
    #x_train=np.multiply(x_train,1./255)
    #val_x=np.multiply(val_x,1./255)
    test_x=np.multiply(test_x,1./255)
    test_y=np.multiply(test_y,1./5)

    y_train=np.multiply(y_train,1./5)
    #dataset changed to 5,0,-5
    val_y=np.multiply(val_y,1./5)
   
    ntrain = len(y_train)
    nval = len(val_y)
    batch_size=30
    print("loading model")
    cnn = cnnModel()
    cnn.summary()
    #cnn.compile(loss = 'binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    #opt=keras.optimizers.Adam(learning_rate=.001)
    print("compiling model")
    cnn.compile(loss = 'mean_squared_error',optimizer=optimizers.SGD(lr=.0001,momentum=0.9),metrics=['acc'])
    #recalse can be None or 1./255
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0, width_shift_range=0,height_shift_range=0, shear_range=0, 
                                       zoom_range=0, horizontal_flip=True)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)
    
    val_generator=val_datagen.flow(val_x,val_y, batch_size=batch_size)
    print("training model")
    history=cnn.fit_generator(train_generator, steps_per_epoch=ntrain//batch_size, epochs=100, validation_data=val_generator,
                            validation_steps=nval//batch_size, use_multiprocessing=True)
    #new version
    #history=cnn.fit(x_train,y_train,steps_per_epoch=ntrain/batch_size, epochs=64, validation_data=val_generator, validation_steps=nval/batch_size)
    print("running model on test set")
    model_score=cnn.evaluate(test_x,test_y)
    print model_score
    #model_predict=cnn.predict(test_x)
    #print model_predict
    #print len(model_predict)
    #print test_y
    #print len(test_y)
    #prediction={'prediction':model_predict,'ground_truth':test_y}
    #prediction_test=pd.DataFrame(model_predict,columns=['prediction'])
    #prediction_test['ground_truth']=test_y
    #prediction_test.to_csv('resNet50_1000epochs.csv')
    
    #test_result=pd.Dataframe.from_dict({'model_predict':model_predict,'ground_truth':test_y})
    
    #test_result.to_csv('test_result.csv')

    cnn.save_weights('/home/sepeagb2/sam/code/abstractResults/03152020/weights/model_weights_resnet50_100epochsSBG_valChange.h5')
    cnn.save('/home/sepeagb2/sam/code/abstractResults/03152020/model/model_resnet50_100epochsSBG_valChange.h5')

def trainModelMulti(modelMulti):
    x_train,y_train,val_x,val_y,test_x,test_y = convertArry()
    ntrain = len(y_train)
    nval = len(val_y)
    batch_size=30
    cnn = modelMulti
    cnn.summary()
    #cnn.compile(loss = 'binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    #opt=adam(lr=1e-3)
    cnn.compile(loss = 'mean_squared_error',optimizer=opt,metrics=['acc'])
            #"""optimizers.SGD(lr=1e-4)""",metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=None, rotation_range=0, width_shift_range=0,height_shift_range=0, shear_range=0, 
                                       zoom_range=0, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=None)
    
    train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)
    val_generator=val_datagen.flow(val_x,val_y, batch_size=batch_size)

    history=cnn.fit_generator(train_generator, steps_per_epoch=ntrain//batch_size, epochs=1000, validation_data=val_generator,
                            validation_steps=nval//batch_size, use_multiprocessing=True)
    #new version
    #history=cnn.fit(x_train,y_train,steps_per_epoch=ntrain/batch_size, epochs=64, validation_data=val_generator, validation_steps=nval/batch_size)
    model_score=cnn.evaluate(test_x,test_y)
    #print model_score
    model_predict=cnn.predict(test_x)
    #print model_predict
    #print len(model_predict)
    #print test_y
    #print len(test_y)
    
    #test_result=pd.Dataframe({'model_predict':model_predict,'ground_truth':test_y})
    
    #test_result.to_csv('test_result.csv')

    cnn.save_weights('/home/sepeagb2/sam/code/saved_weights/model_wieghts_resNet50.h5')
    cnn.save('/home/sepeagb2/sam/code/saved_models/model_resNet50_1000.h5')

def main():
    #ap=argeparse.ArgumentParser()
    #ap.add_argument("-o","--output",required=True, help="path to output plot")
    #ap.add_argument("-g","--gpus",type=int, required=True, help="# of GPUS for training")
    #args=vars(ap.parse_args())
    # grab the number of GPUS and store it in a conveience variable
    #G=args["gpus"]
    #if G <=1:
    trainModel()
    #else:
    #    print("[INFO] training with {} GPUS...".format(G))
    #    with tf.device("/cpu:0"):
            #initialize the model
    #        model=cnnModelMulti()
    #    model=multi_gpu_model(model,gpu=G)
    #    trainModelMulti(model)

    

if __name__=="__main__":    
    main()

