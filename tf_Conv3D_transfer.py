import numpy as np
import h5py
import tensorflow as tf
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
print(tf.__version__)

#initial weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(initial)

#initial bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#convolution layer
def conv3d(x, W):
    return tf.nn.conv3d(input=x, filter=W, strides=[1,1,1,1,1], padding='VALID')

##pooling layer
def max_pool_3d(x):
    return tf.nn.max_pool3d(input=x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

#load data
cube_f_train = h5py.File("data/cube_train_non.hdf5", "r")
cube_f_test = h5py.File("data/cube_test_non.hdf5", "r")
cube_f_valid = h5py.File("data/cube_valid_non.hdf5", "r")

des_f_train = h5py.File("data/des_train.hdf5", "r")
des_f_test = h5py.File("data/des_test.hdf5", "r")
des_f_valid = h5py.File("data/des_valid.hdf5", "r")

cube_train_data = cube_f_train['data']

des_train_data = des_f_train['data']
print(des_train_data.shape)

train_labels = cube_f_train['labels']

scaler = preprocessing.StandardScaler().fit(train_labels)
train_labels_n = scaler.transform(train_labels)

cube_test_data = cube_f_test['data']
cube_valid_data = cube_f_valid['data']
des_test_data = des_f_test['data']
des_valid_data = des_f_valid['data']
test_labels = cube_f_test['labels']
valid_labels = cube_f_valid['labels']

test_labels_n = scaler.transform(test_labels)
valid_labels_n = scaler.transform(valid_labels)

#batch data
def get_batch_data(data, batch, batch_size):
    if data == 'train':
        batch_cube_xs = cube_train_data[batch * batch_size : (batch + 1) * batch_size]
        batch_des_xs = des_train_data[batch * batch_size : (batch + 1) * batch_size]
        batch_ys = train_labels_n[batch * batch_size : (batch + 1) * batch_size]
    elif data == 'test':
        batch_cube_xs = cube_test_data[batch * batch_size : (batch + 1) * batch_size]
        batch_des_xs = des_test_data[batch * batch_size : (batch + 1) * batch_size]
        batch_ys = test_labels_n[batch * batch_size : (batch + 1) * batch_size]
    elif data == 'valid':
        batch_cube_xs = cube_valid_data[batch * batch_size : (batch + 1) * batch_size]
        batch_des_xs = des_valid_data[batch * batch_size : (batch + 1) * batch_size]
        batch_ys = valid_labels_n[batch * batch_size : (batch + 1) * batch_size]
    else:
        batch_cube_xs = ''
        batch_des_xs = ''
        batch_ys = ''
        print('Parameter Error！')
    return batch_cube_xs, batch_des_xs, batch_ys


cube_x = tf.placeholder(tf.float32, [None, 2259404]) 
y = tf.placeholder(tf.float32, [None, 1])

x_data = tf.reshape(cube_x, [-1, 137, 133, 124, 1])  

h_pool0 = max_pool_3d(x_data) 

W_conv1 = weight_variable([5, 5, 5, 1, 4]) 
b_conv1 = bias_variable([4]) 

h_conv1 = tf.nn.relu(conv3d(h_pool0, W_conv1) + b_conv1) 
h_pool1 = max_pool_3d(h_conv1) 


W_conv2 = weight_variable([3, 3, 3, 4, 4]) 
b_conv2 = bias_variable([4])

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2) 
h_pool2 = max_pool_3d(h_conv2) 


W_fc1 = weight_variable([16 * 15 * 14 * 4, 512]) 
b_fc1 = bias_variable([512]) 

h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 15 * 14 * 4])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([512, 128])
b_fc2 = bias_variable([128])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  

des_x = tf.placeholder(tf.float32, [None, 21]) 

des_W_fc1 = weight_variable([21, 10])
des_b_fc1 = bias_variable([10])

des_h_fc1 = tf.nn.relu(tf.matmul(des_x, des_W_fc1) + des_b_fc1)  

#Contacted feature
merge = tf.concat([h_fc2, des_h_fc1], 1)

merge_W_fc1 = weight_variable([138, 64])
merge_b_fc1 = bias_variable([64])

merge_h_fc1 = tf.nn.relu(tf.matmul(merge, merge_W_fc1) + merge_b_fc1) 

merge_W_fc2 = weight_variable([64, 1])
merge_b_fc2 = bias_variable([1])

prediction = tf.matmul(merge_h_fc1, merge_W_fc2) + merge_b_fc2

# RMSE
RMSE = tf.sqrt(tf.reduce_mean(tf.square(y - prediction)))

start_rate = 1e-4
global_steps = tf.Variable(tf.constant(0))
decay_steps = 70
decay_rate = 0.97

# learning rate
learning_rate = tf.train.exponential_decay(start_rate, global_steps, decay_steps, decay_rate, staircase = True)
#using AdamOptimizer to optimize
train_step = tf.train.AdamOptimizer(start_rate).minimize(RMSE)

train_loss_list = []
test_loss_list = []
valid_loss_list = []
epoch_list = []


start = time.time()

batch_size = 2

train_data_length = len(cube_train_data)
n_batch = math.ceil(train_data_length / batch_size)

test_data_length = len(cube_test_data)
test_n_batch = math.ceil(test_data_length / batch_size)

valid_data_length = len(cube_valid_data)
valid_n_batch = math.ceil(valid_data_length / batch_size)

train_prediction_list_batch = np.empty(shape=[1, 80], dtype=np.float32)
test_prediction_list_batch = np.empty(shape=[1, 8], dtype=np.float32)
valid_prediction_list_batch = np.empty(shape=[1, 4], dtype=np.float32)


#with tf.Session(config=tfconfig) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #restore model
    saver = tf.train.Saver()
    model_file='./ckpt/tf_Conv3D.ckpt-174'
    # restore parameters
    saver.restore(sess, model_file)
    
    min_loss = 9
    # training network
    for epoch in range(300):
        for batch in range(n_batch):
            batch_cube_xs, batch_des_xs, batch_ys = get_batch_data('train', batch, batch_size)
            sess.run(train_step, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, y:batch_ys, keep_prob:0.7})
        
        for batch in range(n_batch):
            batch_cube_xs, batch_des_xs, batch_ys = get_batch_data('train', batch, batch_size)
            if batch == 0:
                train_prediction_list = sess.run(prediction, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, keep_prob:1.0})
            else:
                temp = sess.run(prediction, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, keep_prob:1.0})
                train_prediction_list = np.vstack([train_prediction_list, temp])
        
        train_prediction_list = scaler.inverse_transform(train_prediction_list)
        # saving result
        train_prediction_list_batch = np.append(train_prediction_list_batch, train_prediction_list.reshape(1,-1), axis=0)
        # compute loss
        train_loss = np.sqrt(np.mean(np.square(train_labels - train_prediction_list)))
        
        # using testing data to validate model
        for batch in range(test_n_batch):
            batch_cube_xs, batch_des_xs, batch_ys = get_batch_data('test', batch, batch_size)
            if batch == 0:
                test_prediction_list = sess.run(prediction, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, keep_prob:1.0})
            else:
                temp = sess.run(prediction, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, keep_prob:1.0})
                test_prediction_list = np.vstack([test_prediction_list, temp])
       
        test_prediction_list = scaler.inverse_transform(test_prediction_list)
        # saving result
        test_prediction_list_batch = np.append(test_prediction_list_batch, test_prediction_list.reshape(1,-1), axis=0)
        # compute loss
        test_loss = np.sqrt(np.mean(np.square(test_labels - test_prediction_list)))
        
        for batch in range(valid_n_batch):
            batch_cube_xs, batch_des_xs, batch_ys = get_batch_data('valid', batch, batch_size)
            if batch == 0:
                valid_prediction_list = sess.run(prediction, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, keep_prob:1.0})
            else:
                temp = sess.run(prediction, feed_dict={cube_x:batch_cube_xs, des_x:batch_des_xs, keep_prob:1.0})
                valid_prediction_list = np.vstack([valid_prediction_list, temp])
       
        valid_prediction_list = scaler.inverse_transform(valid_prediction_list)
        # saving result
        valid_prediction_list_batch = np.append(valid_prediction_list_batch, valid_prediction_list.reshape(1,-1), axis=0)
        # compute loss
        valid_loss = np.sqrt(np.mean(np.square(valid_labels - valid_prediction_list)))
        
        #saver.save(sess, 'ckpt/tf_Conv3D_TransferLearning_20190527/tf_Conv3D_TransferLearning_20190527.ckpt', global_step=epoch + 1)
        
end = time.time()
print('run_time:',end - start)

#matplotlib inline
plt.figure()
plt.plot(epoch_list, train_loss_list, label='train_loss')
plt.plot(epoch_list, test_loss_list, label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'test_loss'])

cube_f_train.close()
cube_f_test.close()
cube_f_valid.close()
des_f_train.close()
des_f_test.close()
des_f_valid.close()
