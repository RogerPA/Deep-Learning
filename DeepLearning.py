from tf_inumpyut import createFeatureSet 
import tensorflow as tf

import pickle
import numpy

n_labels = 10

batch_size = 100

#Training iterations
iterations = 100

x = tf.placeholder('float')
y = tf.placeholder('float')


keep_rate = 0.7
keep_prob = tf.placeholder(tf.float32)

features, train, feat, label


def create_feature_set(trainingdataset, testdataset):
	training_features = read_file(trainingdataset)
	test_features = read_file(testdataset)

	random.shuffle(training_features)
	training_features = numpy.array(training_features)
	test_features = numpy.array(test_features)
	
	train_feat = list(training_features[:, 0])
	train_label = list(training_features[:, 1])
	test_feat = list(test_features[:, 0])
	test_label = list(test_features[:, 1])


	return train_feat, train_label, test_feat, test_label


def convolutional_network(x):

	weights = {	'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
				'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])), 
				'W_fc': tf.Variable(tf.random_normal([12288, 1024])), 
				'W_out': tf.Variable(tf.random_normal([1024, n_labels]))} 
	biases = {	'b_conv1': tf.Variable(tf.random_normal([32])),  
				'b_conv2': tf.Variable(tf.random_normal([64])),  
				'b_fc': tf.Variable(tf.random_normal([1024])), 
				'b_out': tf.Variable(tf.random_normal([n_labels]))} 
	

	x =	 tf.reshape(x, shape = [-1, 32, 32, 1])

	conv1 = tf.nn.relu(tf.nn.conv2d(x,  weights['W_conv1'], strides =[1, 1, 1, 1],  padding = 'SAME') + biases['b_conv1'])
	conv1 = tf.nn.max_pool(conv1, ksize= [1, 2, 2, 1],  strides = [1, 2, 2, 1],  padding = 'SAME')

	conv2 = tf.nn.relu(tf.nn.conv2d(x,  weights['W_conv2'], strides =[1, 1, 1, 1],  padding = 'SAME')+ biases['b_conv2'])
	conv2 = tf.nn.max_pool(conv2, ksize= [1, 2, 2, 1],  strides = [1, 2, 2, 1],  padding = 'SAME')

	fc = tf.reshape(conv2, [-1, 8*8*64*3])
	fc = tf.nn.relu(tf.matmul(fc,  weights['W_fc'])+biases['b_fc'])

	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc,  weights['W_out'])+biases['b_out']

	return output

def train_neural_network(train_input):
	
	prediction = neural_network_model(train_input)
	cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tensorflow.train.AdamOptimizer().minimize(cost) 

	with tensorflow.Session() as sess:
		sess.run(tensorflow.initialize_all_variables())
	    
		for IT in range(iterations):
			loss = 0
			I=0
			while I < len(train_feat):
				start = I
				end = I + 1
				batch_feat = numpy.array(train_feat[start:end])
				batch_label = numpy.array(train_label[start:end])

				_, K = sess.run([optimizer, cost], feed_dict = {train_input: batch_feat, y: batch_label})
				loss += K
				I += 1
				
			print('Iteration', IT + 1,'loss:',loss)
		
		correct = tensorflow.equal(tensorflow.argmax(prediction, 1), tensorflow.argmax(test_output, 1))
		accuracy = tensorflow.reduce_mean(tensorflow.cast(correct, 'float'))

		print('Accuracy:', accuracy.eval({train_input: test_feat, test_output: test_label}))


train_neural_network(train_input)