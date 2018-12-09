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

class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress

# model parameters
num_classes = 1
max_boxes = 14
Input_shape = 128  # multiple of 32
input_shape = (Input_shape, Input_shape)
threshold = 0.3
ignore_thresh = 0.5

num_epochs = 3
batch_size = 32
momentum = 0.9
decay = 0.0005
learning_rate = 0.001

path = '/home/minh/PycharmProjects'

anchors_paths = PATH + '/model/yolo_anchors.txt'
anchors = read_anchors(anchors_paths)

annotation_path_train = PATH + '/model/boat_train.txt'

data_path_train = PATH + '/model/boat_train.npz'

# Load Data
image_data, box_data, _, y_true = get_training_data(annotation_path_train, data_path_train, input_shape, anchors, 
                                                num_classes, max_boxes, load_previous=True)
### model ###
################################################################################################################
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
Y_ = [Y1, Y2, Y3]
# Calculate loss
loss = compute_loss(scale_total, Y_, anchors, num_classes, print_loss=True)

# Optimizer    
optimizer = tf.train.AdamOptimizer(lr).minimize(loss) #, global_step=global_step)

##################################################################################################################
# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver()

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * batch_size
progress = Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# training loop
num_images = image_data.shape[0]
num_batches = num_images // batch_size
assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
for epoch in range(num_epochs):
    for batch in range(num_batches):
        x = image_data[batch * batch_size:(batch + 1) * batch_size, :]
        y_ = y_true[batch * batch_size:(batch + 1) * batch_size, :]

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, lr: learning_rate}
        _, y = sess.run([optimizer, Y_], feed_dict=feed_dict)

        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)

        # loop state around
        step += batch_size

saved_file = saver.save(sess, 'checkpoints/train') #, global_step=step)
print("Saved file: " + saved_file)