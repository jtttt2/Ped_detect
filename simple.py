import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys
from yolo_utils import get_training_data, read_anchors
from network_function import YOLOv3
from detect_function import predict
from loss_function import compute_loss
tf.set_random_seed(0)

# model parameters
num_classes = 1
max_boxes = 14
Input_shape = 128  # multiple of 32
input_shape = (Input_shape, Input_shape)
threshold = 0.3
ignore_thresh = 0.5

momentum = 0.9
decay = 0.0005
learning_rate = 0.001

path = '/home/minh/PycharmProjects'

anchors_paths = PATH + '/model/yolo_anchors.txt'
anchors = read_anchors(anchors_paths)

annotation_path_train = PATH + '/model/boat_train.txt'

data_path_train = PATH + '/model/boat_train.npz'

# Load Data
x, box_data_train, _, y = get_training_data(annotation_path_train, data_path_train, input_shape, anchors, 
                                                num_classes, max_boxes, load_previous=True)
# model
# tf Graph input
    #graph = tf.Graph()
    #global_step

# input
X = tf.placeholder(tf.float32, [None, 128, 128, 3])
# expected outputs
Y1 = tf.placeholder(tf.float32, shape=[None, Input_shape/32, Input_shape/32, 3, (5+num_classes)])
Y2 = tf.placeholder(tf.float32, shape=[None, Input_shape/16, Input_shape/16, 3, (5+num_classes)])
Y3 = tf.placeholder(tf.float32, shape=[None, Input_shape/8, Input_shape/8, 3, (5+num_classes)])

# Generate output tensor targets for filtered bounding boxes.
scale1, scale2, scale3 = YOLOv3(X, num_classes).feature_extractor()
scale_total = [scale1, scale2, scale3]
# detect
boxes, scores, classes = predict(scale_total, anchors, num_classes, input_shape)

# Label
y_predict = [Y1, Y2, Y3]
# Calculate loss
loss = compute_loss(scale_total, y_predict, anchors, num_classes, print_loss=True)

# Optimizer    
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) #, global_step=global_step)

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver()

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# training loop
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=10):

    # train on one minibatch
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)