#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@author: Yilong Yang
#@time: 2018/03/21 15:58
########################################################
import scipy.io as sio
from scipy import stats
import numpy as np
import pandas as pd
import time
import math
import tensorflow as tf
import sys
import sklearn.preprocessing as preprocessing


CROSS_FOLDS = 5
data_dir = "./EEG_Feature_2Hz/"
label_dir = "./perclos_labels/"

input_height = 17
input_width = 25
input_channel_num = 1
conv_channel_num = 8

learning_rate = 1e-4

def read_dataset(data_dir,label_dir,filename):
    # final_dataset = np.empty([0,2,input_height,input_width])
    final_dataset = np.empty([0,1,input_height,input_width])
    data = sio.loadmat(data_dir + filename)
    labels = sio.loadmat(label_dir + filename)["perclos"]
    dataset_de = np.transpose(data["de_movingAve"], [1, 0, 2])
    dataset_psd = np.transpose(data["psd_movingAve"], [1, 0, 2])
    for i in range(len(dataset_de)):
        temp = np.empty([0,input_height,input_width])
        data_de = preprocessing.scale(dataset_de[i].reshape(-1)).reshape(1,input_height,input_width)
        # data_psd = preprocessing.scale(dataset_psd[i].reshape(-1)).reshape(1,input_height,input_width)
        # temp = np.vstack([temp,data_psd])
        temp = np.vstack([temp,data_de])
        # temp = temp.reshape(1,2,input_height,input_width)
        temp = temp.reshape(1,1,input_height,input_width)
        final_dataset = np.vstack([final_dataset,temp])
    final_dataset = np.transpose(final_dataset,[0,2,3,1])
    return final_dataset,labels

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def conv3d(x, W, kernel_stride,):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv3d(x, W, strides=[1, kernel_stride, kernel_stride, kernel_stride, 1], padding='SAME')


def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name)  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.relu(tf.layers.batch_normalization(conv2d(x, weight, kernel_stride)))

def apply_conv3d(x, filter_depth, filter_height, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_depth, filter_height, filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name) # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    conv_3d = tf.add(conv3d(x, weight, kernel_stride), bias)
    return tf.nn.relu(conv_3d)

def apply_fully_connect(x, x_size, fc_size,name):
    fc_weight = weight_variable([x_size, fc_size],name)
    fc_bias = bias_variable([fc_size],name)
    return tf.nn.relu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size,name):
    readout_weight = weight_variable([x_size, readout_size],name)
    readout_bias = bias_variable([readout_size],name)
    return tf.add(tf.matmul(x, readout_weight), readout_bias)
# kernel parameter
branch1_kernel_height_1st = 5
branch1_kernel_width_1st = 5

branch1_kernel_height_2nd = 4
branch1_kernel_width_2nd = 4

branch1_kernel_height_3rd = 3
branch1_kernel_width_3rd = 3

branch1_kernel_height_4th = 2
branch1_kernel_width_4th = 2

branch2_kernel_height_1st = 5
branch2_kernel_width_1st = 5

branch2_kernel_height_2nd = 4
branch2_kernel_width_2nd = 4

branch2_kernel_height_3rd = 3
branch2_kernel_width_3rd = 3

branch2_kernel_height_4th = 2
branch2_kernel_width_4th = 2

kernel_stride = 1

fc_size = 1024

dropout_prob = 0.5

# training parameter
lambda_loss_amount = 0.5
training_epochs = 200
batch_size = 64

file = sys.argv[1]

dataset,labels = read_dataset(data_dir,label_dir,file)

folds = 5


# input placeholder
cnn_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='cnn_in')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

###########################################################################################
# first CNN layer
conv_1_1 = apply_conv2d(cnn_in, branch1_kernel_height_1st, branch1_kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride,'conv1_1')
print("conv_1_1 shape:", conv_1_1.shape)
# conv_1_2 = apply_conv2d(cnn_in, branch2_kernel_height_1st, branch2_kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride,'conv1_2')
# print("conv_1 shape:", conv_1.shape)
print("\n")
# second CNN layer
conv_2_1 = apply_conv2d(conv_1_1, branch1_kernel_height_2nd, branch1_kernel_width_2nd, conv_channel_num, conv_channel_num * 2, kernel_stride,'conv2_1')
print("conv_2_1 shape:", conv_2_1.shape)
# conv_2_2 = apply_conv2d(conv_1_2, branch2_kernel_height_2nd, branch2_kernel_width_2nd, conv_channel_num, conv_channel_num * 2, kernel_stride,'conv2_2')
# conv_2 = tf.concat([conv_2_1,conv_2_2],3,name="concat2")
# print("conv_2 shape:", conv_2.shape)
print("\n")
# third CNN layer
conv_3_1 = apply_conv2d(conv_2_1, branch1_kernel_height_3rd, branch1_kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4, kernel_stride,'conv3_1')
print("conv_3_1 shape:", conv_3_1.shape)
# conv_3_2 = apply_conv2d(conv_2_2, branch2_kernel_height_3rd, branch2_kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4, kernel_stride,'conv3_2')
# fourth CNN layer
# conv_4_1 = apply_conv2d(conv_3_1, branch1_kernel_height_4th, branch1_kernel_width_4th, conv_channel_num * 4, conv_channel_num * 4, kernel_stride,'conv3_1')
# print("conv_3_1 shape:", conv_4_1.shape)
# conv_4_2 = apply_conv2d(conv_3_2, branch2_kernel_height_4th, branch2_kernel_width_4th, conv_channel_num * 4, conv_channel_num * 4, kernel_stride,'conv3_2')
# conv_3 = tf.concat([conv_3_1,conv_3_2],3,name="concat2")
conv_3 = conv_3_1
print("conv_3 shape:", conv_3.shape)
print("\n")

shape = conv_3.get_shape().as_list()
conv_5_flat = tf.reshape(conv_3, [-1, shape[1] * shape[2] * shape[3]])
cnn_fc = apply_fully_connect(conv_5_flat, shape[1] * shape[2] * shape[3], fc_size,"fc")
print("shape after cnn_full", np.shape(cnn_fc))
# dropout regularizer
# Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
fc_drop = tf.nn.dropout(cnn_fc, keep_prob)

# readout layer
y_ = apply_readout(fc_drop, fc_size, 1,'readout')
# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

# cost function
RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_,Y))))
Pearson = tf.contrib.metrics.streaming_pearson_correlation(y_,Y)
cost = tf.reduce_mean( RMSE+l2, name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# # get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(RMSE, tf.float32), name='accuracy')

#tf.summary.scalar('accuracy',accuracy)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")
# run
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

PREDICTIONS = np.zeros([0])

for curr_fold in range(folds):
    fold_size = dataset.shape[0] // folds
    indexes_list = [i for i in range(len(dataset))]
    indexes = np.array(indexes_list)
    split_list = [i for i in range(curr_fold * fold_size, (curr_fold + 1) * fold_size)]
    split = np.array(split_list)
    cnn_test_x = dataset[split]
    test_y = labels[split]

    split = np.array(list(set(indexes_list) ^ set(split_list)))
    cnn_train_x = dataset[split]
    train_y = labels[split]
    print("training examples:", train_y.shape[0])
    print("test examples	:", test_y.shape[0])
    # set train batch number per epoch
    batch_num_per_epoch = math.floor(cnn_train_x.shape[0] / batch_size) + 1

    # set test batch number per epoch
    accuracy_batch_size = batch_size
    train_accuracy_batch_num = batch_num_per_epoch
    test_accuracy_batch_num = math.floor(cnn_test_x.shape[0] / batch_size) + 1

    print("training samples:", len(train_y))
    print("test samples:", len(test_y))

    batch_num_per_epoch = math.floor(cnn_train_x.shape[0] / batch_size) + 1

    accuracy_batch_size = batch_size
    train_RMSE_batch_num = batch_num_per_epoch
    test_RMSE_batch_num = math.floor(cnn_test_x.shape[0] / accuracy_batch_size) + 1

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        train_RMSE_save = np.zeros(shape=[0], dtype=float)
        test_RMSE_save = np.zeros(shape=[0], dtype=float)
        test_loss_save = np.zeros(shape=[0], dtype=float)
        train_loss_save = np.zeros(shape=[0], dtype=float)
        for epoch in range(training_epochs):
            print("learning rate: ",learning_rate)
            cost_history = np.zeros(shape=[0], dtype=float)
            for b in range(batch_num_per_epoch):
                start = b* batch_size
                if (b+1)*batch_size>train_y.shape[0]:
                    offset = train_y.shape[0] % batch_size
                else:
                    offset = batch_size

                cnn_batch = cnn_train_x[start:(start + offset),:, :, :]

                batch_y = train_y[start:(offset + start), :]
                _, c = session.run([optimizer, cost],
                                   feed_dict={cnn_in: cnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob, phase_train: True})
                cost_history = np.append(cost_history, c)
            if (epoch % 1 == 0):
                train_RMSE = np.zeros(shape=[0], dtype=float)
                test_RMSE = np.zeros(shape=[0], dtype=float)
                test_loss = np.zeros(shape=[0], dtype=float)
                train_loss = np.zeros(shape=[0], dtype=float)

                for i in range(train_RMSE_batch_num):
                    start = i* batch_size
                    if (i+1)*batch_size>train_y.shape[0]:
                        offset = train_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    #offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size)
                    train_cnn_batch = cnn_train_x[start:(start + offset), :, :, :]
                    train_batch_y = train_y[start:(start + offset), :]

                    train_a, train_c = session.run([accuracy, cost],
                                                   feed_dict={cnn_in: train_cnn_batch,Y: train_batch_y, keep_prob: 1.0, phase_train: False})

                    train_loss = np.append(train_loss, train_c)
                    train_RMSE = np.append(train_RMSE, train_a)
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                      np.mean(train_loss), "Training RMSE: ", np.mean(train_RMSE))
                train_RMSE_save = np.append(train_RMSE_save, np.mean(train_RMSE))
                train_loss_save = np.append(train_loss_save, np.mean(train_loss))

                for j in range(test_RMSE_batch_num):
                    start = j * batch_size
                    if (j+1)*batch_size>test_y.shape[0]:
                        offset = test_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    #offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                    test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :]
                    test_batch_y = test_y[start:(offset + start), :]

                    test_a, test_c = session.run([accuracy, cost],
                                                 feed_dict={cnn_in: test_cnn_batch,Y: test_batch_y,keep_prob: 1.0, phase_train: False})

                    test_RMSE = np.append(test_RMSE, test_a)
                    test_loss = np.append(test_loss, test_c)

                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                      np.mean(test_loss), "Test RMSE: ", np.mean(test_RMSE), "\n")
                test_RMSE_save = np.append(test_RMSE_save, np.mean(test_RMSE))
                test_loss_save = np.append(test_loss_save, np.mean(test_loss))


            # reshuffle
            index = np.array(range(0, len(train_y)))
            np.random.shuffle(index)
            cnn_train_x=cnn_train_x[index]
            train_y=train_y[index]


            # learning_rate decay
            if (np.mean(train_RMSE) > 0.15):
                learning_rate = 1e-4
            elif (0.1 < np.mean(train_RMSE) < 0.15):
                learning_rate = 5e-5
            elif ( np.mean(train_RMSE) < 0.1):
                learning_rate = 5e-6



        test_RMSE = np.zeros(shape=[0], dtype=float)

        for k in range(test_RMSE_batch_num):
            start = k * batch_size
            if (k+1)*batch_size>test_y.shape[0]:
                offset = test_y.shape[0] % batch_size
            else:
                offset = batch_size
            #offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
            test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :]
            test_batch_y = test_y[start:(offset + start), :]

            test_a, test_c = session.run([accuracy, cost],
                                                         feed_dict={cnn_in: test_cnn_batch,Y: test_batch_y, keep_prob: 1.0, phase_train: False})
            test_t = test_batch_y

            test_RMSE = np.append(test_RMSE, test_a)
            test_loss = np.append(test_loss, test_c)

        print("(" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
              "Final Test RMSE: ", np.mean(test_RMSE))
        # save result
    #    os.system("mkdir -p ./result/cnn_rnn_parallel/tune_rnn_layer/" + output_dir)
        prediction = session.run(y_,feed_dict={cnn_in: cnn_test_x, Y: test_y, keep_prob: 1 - dropout_prob, phase_train: False})
        print("prediction shape:",len(prediction),"---","test_y shape:",len(test_y))

        result = pd.DataFrame(
            {'epoch': range(1, epoch + 2), "train_RMSE": train_RMSE_save, "test_RMSE": test_RMSE_save,
             "train_loss": train_loss_save, "test_loss": test_loss_save})
        RMSE = pd.DataFrame(
            {"prediction:":np.squeeze(prediction), "ground_truth:": np.squeeze(test_y)}
        )
        writer = pd.ExcelWriter(
            "./result/"+ file +str(curr_fold)+ ".xlsx")
        result.to_excel(writer, 'result', index=False)
        RMSE.to_excel(writer,"RMSE",index=False)
        
        writer.save()

        PREDICTIONS = np.append(PREDICTIONS,prediction)
        # save model
        '''
        saver = tf.train.Saver()
        saver.save(session,
                   "./result/cnn_rnn_parallel/tune_rnn_layer/" + output_dir + "/model_" + output_file)
        '''

writer = pd.ExcelWriter("./result/"+file+".xlsx")
PREDICTIONS = np.squeeze(PREDICTIONS)
labels = np.squeeze(labels)

result = pd.DataFrame({"prediction:":PREDICTIONS,"labels:":labels})
result.to_excel(writer,"result",index=False)

RMSE = np.sqrt(np.mean(np.square(PREDICTIONS-labels)))
COR,p_value = stats.pearsonr(PREDICTIONS,labels)

print("RMSE:",RMSE)
print("COR:",COR)

matrics = pd.DataFrame({"RMSE:":RMSE,"COR:":COR},index=[0])
matrics.to_excel(writer,"matrics",index=False)

writer.save()

print("**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN End **********\n")