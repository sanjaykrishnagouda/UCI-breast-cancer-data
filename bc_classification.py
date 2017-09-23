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

def main():
	data = load("bc.csv")
	
	x_train = data[0]
	y_train = data[1]
	x_test = data[2]
	y_test = data[3]

	x = tf.placeholder(tf.float32, [None,19])
	W = tf.Variable(tf.random_normal([19,5])/np.sqrt(455))
	b = tf.Variable(tf.random_normal([5])*0.01)
	y = tf.nn.softmax(tf.matmul(x,W)+b)
	y_ = tf.placeholder(tf.float32, [None, 10])
	
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices = [1]))
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()


	for el in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(50)
		sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
		
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


main()