import pandas as pd
import numpy as np
import sklearn.ensemble
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import time, sys
from operator import itemgetter
import tensorflow as tf

def load(path):
	complete_data = pd.read_csv(str(path), header = 0)
	x = df.ix[:,df.columns!='diagnosis']
	y = df.ix[:,df.columns=='diagnosis']
	y = y['diagnosis'].map({'M':1,'B':0})
	
	x = x.drop(['id','Unnamed: 32','area_mean','perimeter_mean','concavity_mean','concave points_mean','area_worst','perimeter_worst',
		'concave points_worst','concavity_worst','area_se','perimeter_se'],axis = 1)
	features = []
	for i in x:
		features.append(i)
	frames = [x,y]
	total = pd.concat(frames,axis = 1)
	del features[-1]
	train, test = train_test_split(frames, test_size = 0.2)
	train_x,test_x = train[features],test[features]
	train_y, test_y = train.diagnosis, test.diagnosis
	data = [train_x,train_y,test_x,test_y]
	return data

	def NeuralNetwork(x,features):
	# Neural Network with one hidden layer
	# Will try and work with more hidden layers
	# Optimization of hyperparameters for one hidden layer NN supported
	# WORK IN PROGRESS
	del features[-1]
	train, test = train_test_split(x, test_size = 0.2)
	train_x,test_x = train[features],test[features]
	train_y, test_y = train.diagnosis.values.reshape((train.diagnosis.shape[0],1)),test.diagnosis.values.reshape((test.diagnosis.shape[0],1))
	train_y = train_y.T
	test_y = test_y.T
	
	
	# DEFINING PARAMETERS 
	n_h = 5 # number of hidden units
	
	# layer 1 (input layer)

	w1 = np.random.randn(n_h,train_x.shape[1])*0.01/np.sqrt(train_x.shape[0])
	b1 = np.random.randn(n_h,1)*0.01
	w2 = np.random.randn(train_y.shape[0],n_h)*0.01/np.sqrt(train_y.shape[0])
	b2 = np.random.randn(train_y.shape[0],1)*0.01

	num_iters = 10
	learning_rate = 0.1
	m = (1/train_y.shape[1])
	for i in range(num_iters):
		#print(w1.shape,w2.shape,b1.shape,b2.shape) # checkind dimensions

		###### FORWARD PROPAGATION #######

		z1 = np.dot(w1,train_x.T) + b1 # linear transform
		A1 = np.maximum(z1,0.001*z1) # ReLu Activation or maybe we can use leaky ReLu 
		z2 = np.dot(w2,A1) + b2
		A2 = (1/(1+np.exp(-z2))) # sigmoid activation to get y hat 

		
		#print(np.log(abs(1-A2)))
		#cost = -(1/m)*np.sum(np.dot(train_y,np.log(A2).T)+np.dot(1-train_y,np.log(1-A2).T))
		cost = log_loss(train_y,A2) # this is throwing some error, 
									 # no patiance to workaround, will write own loss function
		#print(z2)
		#print("Iteration %i, cost %.4f"%(i,cost))

		

		###### 	BACKWARD PROPAGATION #######

		#dA2 = train_y/A2 + (1-train_y)/(1-A2)

		dA2 = - (np.divide(train_y, A2) - np.divide(1 - train_y, 1 - A2))
		#print(dA2)
		dz2 = (dA2*(z2*(1-z2))) # deriative of sigmoid 

		dw2 = (np.dot(dz2,A1.T))/m

		# print(w2.shape,dw2.shape) # checking fwd bwd param dims

		db2 = np.sum(dz2, axis = 1, keepdims = True) / m

		dA1 = np.dot(w2.T,dz2)

		
		dz1 = (dA1 ) #relu derivative (modified, ReLu is not differentiable at 0 )

		dw1 = np.dot(dz1,train_x) / m

		db1 = np.sum(dz1, axis = 1, keepdims = True) / m
		#print(w1.shape,dw1.shape, b1.shape, db1.shape) # checking fwd bwd param dims
		
		##### UPDATING PARAMETERS ######

		w1 = w1 - learning_rate * dw1
		b1 = b1 - learning_rate * db1
		w2 = w2 - learning_rate * dw2
		b2 = b2 - learning_rate * db2
		#print("Iteration %i, cost2 %.4f"%(i,cost2))