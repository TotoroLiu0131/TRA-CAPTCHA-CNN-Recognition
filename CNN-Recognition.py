import theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation ,Convolution2D , MaxPooling2D , Flatten
from keras.optimizers import SGD,Adam
from keras.models import load_model
import os
from PIL import Image 
import numpy as np
from numpy import  *  
from keras.constraints import maxnorm
from keras.utils import np_utils
from datetime import date, timedelta,datetime
import cv2

trainingDataPath = './TRAticket/model'
modelPath = trainingDataPath+'/Keras'
testingdataPath = './TRAticket/testing_data/'
tempPath = './TRAticket/template/'
captchaOutput ='output.jpg'

def buildModel():
	
	data = []
	target = []
	# todo : read image 
	for index in  xrange(0,10):
		folderPath = os.path.join(trainingDataPath,str(index))
		for file in os.listdir(folderPath) :
			filePath = os.path.join(folderPath,file)
			if file.endswith('.jpg') :
				im = cv2.imread(filePath)
				arr = np.array(im.reshape(3,8,8))
				data.append(arr)
				target.append(int(index))

	X_train = np.array(data)
	Y_train = np.array(target)

	X_train = X_train.astype('float32')
	X_train = X_train / 255.0


	Y_train = np_utils.to_categorical(Y_train)

	num_classes = Y_train.shape[1]

	# todo : build model
	model = Sequential()

	# todo : Conv layer 1 output shape (3,8,8)
	model.add(Convolution2D(32, 4, 4, input_shape=(3,8,8), border_mode='same', 
		activation='relu', W_constraint=maxnorm(3)))

	# todo : Pooling layer 1 output shape (3,4,4)
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# todo : Conv layer 2 output shape (6,4,4)
	model.add(Convolution2D(32, 4, 4, border_mode='same',
		activation='relu', W_constraint=maxnorm(3)))

	# todo : Pooling layer 2 output shape (6,2,2)
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# todo : Fully connected layer 1 input shape output shape
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.2))

	# todo : Fully connected layer 2 input shape , output shape(10) for 10 classes
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	# todo : fit and compile model
	epochs = 25
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, 
		metrics=["accuracy","precision","recall","fmeasure"])
	print(model.summary())
	model.fit(X_train, Y_train,nb_epoch=epochs, batch_size=32)

	# todo : save model
	model.save(modelPath+'/twTicket_model.h5')

def predict_class():
	try:
		data =[]
	
		for file in os.listdir(tempPath) :
			if not file.endswith('.jpg') or file == captchaOutput :
				continue 
			filePath = os.path.join(tempPath,file)
			im = cv2.imread(filePath) 
			arr = np.array(im)
			data.append(arr)

		X_pred = np.array(data)
		X_pred = X_pred.astype('float32')
		X_pred = X_pred / 255.0

		# todo : load model
		model = load_model(modelPath+'/twTicket_model.h5')

		predicted = model.predict_classes(X_pred, batch_size=32, verbose=0)
		print predicted
	except Exception, e:
		print e

def predict_evaluate():

	try:	
		# todo : load testing data 
		data = []
		target = []

		for index in  xrange(0,10):
			folderPath = os.path.join(testingdataPath,str(index))
			for file in os.listdir(folderPath) :
				filePath = os.path.join(folderPath,file)
				if file.endswith('.jpg') :
					im = cv2.imread(filePath)
					arr = np.array(im.reshape(3,8,8))
					data.append(arr)
					target.append(int(index))

		X_test = np.array(data)
		X_test = X_test.astype('float32')
		X_test = X_test / 255.0

		Y_test = np.array(target)
		Y_test = np_utils.to_categorical(Y_test)

		num_classes = Y_test.shape[1]

		# todo : load model
		model = load_model(modelPath+'/twTicket_model.h5')
		
		loss ,accuracy,precision,recall,fmeasure = model.evaluate(X_test,Y_test)
		print ('\nresult')
		print 'loss ' + str(loss)
		print 'accuracy ' + str(accuracy)
		print 'precision ' + str(precision)
		print 'recall ' + str(recall)
		print 'fmeasure ' + str(fmeasure)
	
	except Exception, e:
		print e

	

if __name__ == '__main__':

	start_time=datetime.now() 

	# buildModel()
	predict_class()
	predict_evaluate()

	finish_time=datetime.now()

	print 'Starting time: '+ start_time.strftime('%Y-%m-%d %H:%M:%S')
	print

	print 'finish time: '+ finish_time.strftime('%Y-%m-%d %H:%M:%S')

	print

	print 'total time: '+ str(finish_time-start_time)


