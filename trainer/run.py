from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import subprocess
import tempfile
import time
import numpy as np
import random
import argparse

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

try:
    from tensorflow.python.ops import ctc_ops
except ImportError:
    from tensorflow.contrib.ctc import ctc_ops
    
from tensorflow.contrib import grid_rnn

from tensorflow.python.util import nest
from math import sqrt
from model import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.8, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10, 'Number of steps to run trainer.')
flags.DEFINE_integer('display_step', 5, 'Number of steps after one display.')
flags.DEFINE_integer('num_classes', 29, 'Number of classes.')
flags.DEFINE_integer('save_step', 1, 'Number of steps after to save model.')
flags.DEFINE_integer('layers', 2, 'Number of layers.')
flags.DEFINE_integer('hidden', 100, 'Number of units in hidden layer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('rnn_cell', 'LSTM', 'rnn cell to use')
flags.DEFINE_string('rnn_type', 'bidirectional', 'rnn type propagation to use')
flags.DEFINE_string('model_dir', 'models', 'Directory to save the model.')
#gs://my-first-bucket-mosnoi/handwritten/m1/
flags.DEFINE_string('board_path', 'TFboard', 'Directory to save board data ')
flags.DEFINE_string('input_path_test', 'data/handwritten-test-16.tfrecords','get data for testing')
flags.DEFINE_string('input_path', 'data/handwritten-test-16.tfrecords',
                    'get data for training, if filenameNr>1 the input data '
                    'should have {} in order to farmat it, and get more than one'
                    'file for training ')



def fast_run(args):
    model = Model(args)
    feed = {}
    #feed[model.train_batch]=False
    #xx,ss,yy=model.inputs(args.input_path)
    model.graphMaker()
    #sess = tf.Session()
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #tf.train.start_queue_runners(sess=sess)
    for var in tf.trainable_variables():
            print(var.op.name)
    #t=model.sess.run(model.ttt)
    #print(t.shape)
    return
    #print(yyy[1])
    print('len:',xxx.shape)
    import matplotlib.cm as cm
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,4))
    #plt.imshow()
    plt.imshow(np.asarray(xxx[0]).reshape((36,90))+0.5, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.savefig("img.jpg")
    plt.clf() ; plt.cla()
    
def main(_):
    parser = argparse.ArgumentParser()

    #run param
    parser.add_argument('--train', dest='train', action='store_true', help='train the model')
    parser.add_argument('--sample', dest='train', action='store_false', help='sample from the model')
    parser.add_argument('--gpu', dest='gpu', action='store_false', help='train the model with gpu')
    parser.add_argument('--distributed', dest='distributed', action='store_false', help='train the model with gpu')
    
    #general model params
    parser.add_argument('--insertLastState', dest='insertLastState', action='store_true', help='insert last state')
    parser.add_argument('--hidden', type=int, default=FLAGS.hidden, help='size of RNN hidden state')
    parser.add_argument('--layers', type=int, default=FLAGS.layers, help='number of layers')
    parser.add_argument('--rnn_cell', type=str, default=FLAGS.rnn_cell, help='rnn cell to use')
    parser.add_argument('--keep_prob', type=float, default=FLAGS.keep_prob, help='prob. of keeping neuron during dropout')
    parser.add_argument('--learning_rate', type=float, default=FLAGS.learning_rate, help='initial learning_rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for RMSP optimizer')
    parser.add_argument('--decay', type=float, default=0.95, help='decay for RMSP optimizer')
    parser.add_argument('--ctc_decoder', type=str, default='greedy', help='ctc_decoder value for ctc loss')
    parser.add_argument('--optimizer', type=str, default="ADAM", help='optimizer to use')
    parser.add_argument('--initializer', type=str, default="graves", help='initializer to use')
    parser.add_argument('--bias', type=float, default=0.1, help='initializer to use for bias')
    
    #iteration param
    parser.add_argument('--save_step', type=int, default=FLAGS.save_step, help='after how many iteration to save the step')
    parser.add_argument('--max_steps', type=int, default=FLAGS.max_steps, help='number of iterations')
    parser.add_argument('--display_step', type=int, default=FLAGS.display_step, help='number of iterations to display after')
    parser.add_argument('--batch_size', type=int, default=FLAGS.batch_size, help='the size of the batch')
    parser.add_argument('--shuffle_batch', dest='shuffle_batch', action='store_true', help='train the model')
    
    #files param
    parser.add_argument('--model_dir', type=str, default=FLAGS.model_dir, help='location to save model')
    parser.add_argument('--input_path', type=str, default=FLAGS.input_path, help='location to get data for training')
    parser.add_argument('--input_path_test', type=str, default=FLAGS.input_path_test, help='location to get data for testing')
    parser.add_argument('--board_path', type=str, default=FLAGS.board_path, help='location to save statistics')
    parser.add_argument('--filenameNr', type=int, default=1, help='if more than one use format(i) to make the files input')
    
    #static param
    parser.add_argument('--width', type=int, default=90, help='image width')
    parser.add_argument('--height', type=int, default=36, help='image height')
    parser.add_argument('--num_classes', type=int, default=FLAGS.num_classes, help='number of classes')
    
    #defaults
    parser.set_defaults(train=True)
    parser.set_defaults(insertLastState=False)
    parser.set_defaults(gpu=False)
    parser.set_defaults(shuffle_batch=False)
    parser.set_defaults(distributed=False)
    args = parser.parse_args()
    if False:
        fast_run(args)
        return
    run_training(args) if args.train else run_sample(args)


if __name__ == '__main__':
    tf.app.run()
