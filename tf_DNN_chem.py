import numpy as np
import h5py
import tensorflow as tf
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math

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

#pooling layer
def max_pool_3d(x):

    return tf.nn.max_pool3d(input=x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

#load data
des_f_train = h5py.File("data/des_train.hdf5", "r")
des_f_test = h5py.File("data/des_test.hdf5", "r")
des_f_valid = h5py.File("data/des_valid.hdf5", "r")

# des training
des_train_data = des_f_train['data']
# training label
train_labels = des_f_train['labels']
# normalization
scaler = preprocessing.StandardScaler().fit(train_labels)
train_labels_n = scaler.transform(train_labels)

# des testing
des_test_data = des_f_test['data']
# des validation
des_valid_data = des_f_valid['data']
# testing label
test_labels = des_f_test['labels']
# validation label
valid_labels = des_f_valid['labels']
#  normalization
test_labels_n = scaler.transform(test_labels)
valid_labels_n = scaler.transform(valid_labels)

# batch data
def get_batch_data(data, batch, batch_size):
    if data == 'train':
        batch_des_xs = des_train_data[batch * batch_size : (batch + 1) * batch_size]
        batch_ys = train_labels_n[batch * batch_size : (batch + 1) * batch_size]
    elif data == 'test':  
        batch_des_xs = des_test_data[batch * batch_size : (batch + 1) * batch_size]
        batch_ys = test_labels_n[batch * batch_size : (batch + 1) * batch_size]
    else:
        batch_des_xs = des_valid_data[batch * batch_size : (batch + 1) * batch_size]
        batch_ys = valid_labels_n[batch * batch_size : (batch + 1) * batch_size]
    return  batch_des_xs, batch_ys

des_x = tf.placeholder(tf.float32, [None, 21])

# initial the first-layer weight
des_W_fc1 = weight_variable([21, 10])
des_b_fc1 = bias_variable([10])

des_h_fc1 = tf.nn.relu(tf.matmul(des_x, des_W_fc1) + des_b_fc1)  

# initial the second-layer weight
des_W_fc2 = weight_variable([10, 1])
des_b_fc2 = bias_variable([1])

#computing output
prediction = tf.matmul(des_h_fc1, des_W_fc2) + des_b_fc2 

y = tf.placeholder(tf.float32, [None, 1])
# RMSE
RMSE = tf.sqrt(tf.reduce_mean(tf.square(y - prediction)))

start_rate = 1e-4
global_step = tf.Variable(tf.constant(0))
decay_steps = 70
decay_rate = 0.97

# learning rate
learning_rate = tf.train.exponential_decay(start_rate, global_step, decay_steps, decay_rate, staircase = True)

#using AdamOptimizer to optimize
train_step = tf.train.AdamOptimizer(learning_rate).minimize(RMSE)

#saving result
train_loss_list = []
test_loss_list = []
valid_loss_list = []
epoch_list = []

start = time.time()


batch_size = 2

train_data_length = len(des_train_data)
n_batch = math.ceil(train_data_length / batch_size)

test_data_length = len(des_test_data)
test_n_batch = math.ceil(test_data_length / batch_size)

valid_data_length = len(des_valid_data)
valid_n_batch = math.ceil(valid_data_length / batch_size)

# traning prediction
train_prediction_list_batch = np.empty(shape=[1, 848], dtype=np.float32)

# testing prediction
test_prediction_list_batch = np.empty(shape=[1, 289], dtype=np.float32)

# validation prediction
valid_prediction_list_batch = np.empty(shape=[1, 29], dtype=np.float32)

#with tf.Session(config=tfconfig) as sess:
with tf.Session( ) as sess:
    sess.run(tf.global_variables_initializer())
    
    # construct saver object
    saver = tf.train.Saver()   # default max_to_keep=5
    
    min_loss = 9
    # training the network
    for epoch in range(300):
        for batch in range(n_batch):
            batch_des_xs, batch_ys = get_batch_data('train', batch, batch_size)
            sess.run(train_step, feed_dict={des_x:batch_des_xs, y:batch_ys})
        
        for batch in range(n_batch):
            batch_des_xs, batch_ys = get_batch_data('train', batch, batch_size)
            if batch == 0:
                train_prediction_list = sess.run(prediction, feed_dict={ des_x:batch_des_xs})
            else:
                temp = sess.run(prediction, feed_dict={ des_x:batch_des_xs})
                train_prediction_list = np.vstack([train_prediction_list, temp])
    
        train_prediction_list = scaler.inverse_transform(train_prediction_list)
        # saving result
        train_prediction_list_batch = np.append(train_prediction_list_batch, train_prediction_list.reshape(1,-1), axis=0)
        # computing the loss
        train_loss = np.sqrt(np.mean(np.square(train_labels - train_prediction_list)))
        
        # computing the loss
        for batch in range(test_n_batch):
            batch_des_xs, batch_ys = get_batch_data('test', batch, batch_size)
            if batch == 0:
                test_prediction_list = sess.run(prediction, feed_dict={ des_x:batch_des_xs })
            else:
                temp = sess.run(prediction, feed_dict={ des_x:batch_des_xs })
                test_prediction_list = np.vstack([test_prediction_list, temp])
       
        test_prediction_list = scaler.inverse_transform(test_prediction_list)
        test_prediction_list_batch = np.append(test_prediction_list_batch, test_prediction_list.reshape(1,-1), axis=0)
        # compute the loss
        test_loss = np.sqrt(np.mean(np.square(test_labels - test_prediction_list)))
        
        # estimate the model
        for batch in range(valid_n_batch):
            batch_des_xs, batch_ys = get_batch_data('valid', batch, batch_size)
            if batch == 0:
                valid_prediction_list = sess.run(prediction, feed_dict={ des_x:batch_des_xs })
            else:
                temp = sess.run(prediction, feed_dict={ des_x:batch_des_xs })
                valid_prediction_list = np.vstack([valid_prediction_list, temp])
        
        valid_prediction_list = scaler.inverse_transform(valid_prediction_list)
       
        valid_prediction_list_batch = np.append(valid_prediction_list_batch, valid_prediction_list.reshape(1,-1), axis=0)
        # compute loss
        valid_loss = np.sqrt(np.mean(np.square(valid_labels - valid_prediction_list)))
    
        # saving model 
        if valid_loss < min_loss:
            min_loss = valid_loss
            saver.save(sess, 'ckpt_des/tf_DNN_des.ckpt', global_step=epoch + 1)
        
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        valid_loss_list.append(valid_loss)
        epoch_list.append(epoch+1)
        print('Iter ' + str(epoch+1) + ', Testing loss=' + str(test_loss)+ ', Training loss=' + str(train_loss)+ ', Valid loss=' + str(valid_loss))

end = time.time()
print('run_time:',end - start)

#matplotlib inline
plt.figure() 
plt.plot(epoch_list, train_loss_list, label='train_loss')
plt.plot(epoch_list, test_loss_list, label='test_loss')
plt.plot(epoch_list, valid_loss_list, label='valid_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'test_loss','valid_loss'])
plt.savefig("result/fig_des.jpg")


des_f_train.close()
des_f_test.close()