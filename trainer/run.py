from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
flags.DEFINE_string('input_path_test', 'data/handwritten-test.tfrecords','get data for testing')
flags.DEFINE_string('input_path', 'data/handwritten-test.tfrecords',
                    'get data for training, if filenameNr>1 the input data '
                    'should have {} in order to farmat it, and get more than one'
                    'file for training ')


class Model(object):
  def __init__(self, args):
    """ init the model with hyper-parameters etc """
    self.height, self.width = args.height, args.width
    self.num_hidden = args.hidden
    self.num_classes = args.num_classes
    self.learning_rate = args.learning_rate
    self.input_path  = args.input_path
    self.batch_size = args.batch_size
    self.layers = args.layers
    self.hidden = args.hidden
    self.model_dir = args.model_dir
    self.board_path = args.board_path
    self.keep_prob = args.keep_prob
    self.filenameNr = args.filenameNr
    self.input_path_test = args.input_path_test
    self.gpu = args.gpu
    self.insertLastState = args.insertLastState
    self.rnn_cell = args.rnn_cell
    self.optimizer = args.optimizer
    self.train_b = args.train
    self.momentum = args.momentum
    self.decay = args.decay
    self.initializer = args.initializer
    self.bias = args.bias
    print(args)
    
  def weight_variable(self,shape,name="v"):
        if self.initializer == "graves" and False:
            initial = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
        else:
            initial = tf.truncated_normal(shape, stddev=.075)
        return tf.Variable(initial,name=name+"_weight")

  def bias_variable(self,shape,name="v"):
   initial = tf.constant(self.bias, shape=shape)
   return tf.Variable(initial, name=name+"_bias")

  def conv2d(self,x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self,x,h=2,w=1):
      return tf.nn.max_pool(x, ksize=[1, h, w, 1],
                        strides=[1, h, w, 1], padding='SAME')

  def convLayer(self,data,chanels_out,size_window=5,keep_prob=0.8,maxPool=False,scopeN="l1"):
    """Implement convolutional layer
    @param data: [batch,h,w,chanels]
    @param chanels_out: number of out chanels
    @param size_windows: the windows size
    @param keep_prob: the dropout amount
    @param maxPool: if true the max pool is applyed
    @param scopeN: the scope name
    
    returns convolutional output [batch,h,w,chanels_out]
    """
    with tf.name_scope("conv-"+scopeN):
        shape = data.get_shape().as_list()
        with tf.variable_scope("convVars-"+scopeN) as scope:
            W_conv1 = self.weight_variable([size_window, size_window, shape[3], chanels_out], scopeN)
            b_conv1 = self.bias_variable([chanels_out], scopeN)
        h_conv1 = tf.nn.relu(self.conv2d(data, W_conv1) + b_conv1)
        if keep_prob and keep_prob!=1 and self.train_b:
            h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
        if maxPool:
            h_conv1 = self.max_pool_2x2(h_conv1)
    return h_conv1



  def ctc_loss(self,outputs, targets, seq_len, num_classes,initial_learning_rate, keep_prob=0.8, scopeN="l1-ctc_loss"):
    """Implements ctc loss
    
    @param outputs: [batch,h,w,chanels]
    @param targets: sparce tensor 
    @param seq_len: the length of the inputs sequences [batch]
    @param num_classes: the number of classes
    @param initial_learning_rate: learning rate
    @param keep_prob: if true dropout layer
    @param scopeN: the scope name
    
    @returns: list with [optimizer, cost, Inaccuracy- label error rate, decoded output of the batch]
    """
    with tf.name_scope('Train'):
        with tf.variable_scope("ctc_loss-"+scopeN) as scope:
            W = tf.Variable(tf.truncated_normal([self.hidden*2,
                                                 num_classes],
                                                stddev=0.1))
            # Zero initialization
            b = tf.Variable(tf.constant(0., shape=[num_classes]))

        tf.summary.histogram('histogram-b-ctc', b)
        tf.summary.histogram('histogram-w-ctc', W)

        # Doing the affine projection
        logits = tf.matmul(outputs, W) +  b 

        if keep_prob is not None:
            logits = tf.nn.dropout(logits, keep_prob)

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [ self.batch_size,self.width, num_classes])    
        logits =  tf.transpose(logits, [1,0,2])

        with tf.name_scope('CTC-loss'):
            loss = ctc_ops.ctc_loss(logits, targets, seq_len)
            cost = tf.reduce_mean(loss)
            
        with tf.name_scope('Optimizer'):
            if self.optimizer == "ADAM":
                optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,name="AdamOptimizer").minimize(cost)
            elif self.optimizer == "RMSP":
                optimizer = tf.train.RMSPropOptimizer(learning_rate=initial_learning_rate, decay=self.decay, momentum=self.momentum).minimize(cost)
            else:
                raise Exception("model type not supported: {}".format(self.optimizer))
        
        with tf.name_scope('Prediction'):
            #decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_len)
            decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_len)

            # Inaccuracy: label error rate
            ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets))
    return optimizer, cost, ler, decoded
    
  def read_and_decode_single_example(self,filename,test=False):
    with tf.name_scope('TFRecordReader'):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down
        files = [filename] if self.filenameNr==1 or test else [filename.format(i) for i in range(self.filenameNr)]
        filename_queue = tf.train.string_input_producer(files,
                                                        num_epochs=None)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'seq_len': tf.FixedLenFeature([1], tf.int64),
                'target': tf.VarLenFeature(tf.int64),     
                'imageInput': tf.FixedLenFeature([self.height*self.width], tf.float32)
            })
        # now return the converted data
        imageInput = features['imageInput']
        seq_len     = features['seq_len']
        target     = features['target']
    return imageInput, seq_len , target

  def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    
  def inputs(self,fileI,test=False):
    with tf.name_scope('batch'):
        imageInput, seq_len , target = self.read_and_decode_single_example(fileI,test)
        
        imageInputs, seq_lens , targets = tf.train.batch([imageInput,seq_len,target], batch_size=self.batch_size)
        
        imageInputs  = tf.cast(imageInputs, tf.float32)
        seq_lens = tf.cast(seq_lens, tf.int32)      
        targets = tf.cast(targets, tf.int32)      
        seq_lens = tf.reshape(seq_lens,[self.batch_size])             
        
        imageInputs = tf.reshape(imageInputs , [self.batch_size,self.height, self.width,1])
        
        return imageInputs, seq_lens, targets
    
  def inference(self, batch_x):
       
        [imageInputs, seq_len] = batch_x
        tf.summary.image("images", imageInputs)
        with tf.name_scope('convLayers'):
            conv1 = self.convLayer(imageInputs, 32 ,              scopeN="l1",keep_prob=self.keep_prob,maxPool=True)
            conv2 = self.convLayer(conv1,       64 ,              scopeN="l2",keep_prob=self.keep_prob,maxPool=True)
            conv3 = self.convLayer(conv2,      128 ,size_window=3,scopeN="l3",keep_prob=self.keep_prob,maxPool=False)
            conv4 = self.convLayer(conv3,      256 ,size_window=3,scopeN="l4",keep_prob=self.keep_prob,maxPool=False)
        
        with tf.name_scope('preprocess'):
            hh,ww,chanels = conv4.get_shape().as_list()[1:4]
            assert ww == self.width,'image does not have to become smaller in width'
            assert chanels == 256
            
            h_pool2_flat = tf.transpose(conv4, [2, 0, 1, 3])
            h_pool2_flat = tf.reshape(h_pool2_flat, [ww, self.batch_size ,hh*chanels])
            
            # Permuting batch_size and n_steps
            #x = tf.transpose(h_pool2_flat, [2, 0, 1])
            # Reshape to (n_steps*batch_size, n_input)
            x = tf.reshape(h_pool2_flat, [-1, hh*chanels])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.split(0, ww, x)
            
        with tf.name_scope('BRNN'):
            if self.initializer == "graves":
                myInitializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
            else:
                myInitializer = tf.truncated_normal(shape, stddev=0.1)
                
            if self.rnn_cell == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(self.hidden, state_is_tuple=True,initializer=myInitializer)
            elif self.rnn_cell == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(self.hidden)
            elif self.rnn_cell == "GRIDLSTM":
                cell = grid_rnn.Grid2LSTMCell(self.hidden,use_peepholes=True,forget_bias=1.0)
            elif self.rnn_cell == "GRIDGRU":
                cell = grid_rnn.Grid2GRUCell(self.hidden)
            else:
                raise Exception("model type not supported: {}".format(self.rnn_cell))
            if self.keep_prob!=1 and self.train_b:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)

            stackf = tf.nn.rnn_cell.MultiRNNCell([cell] * (self.layers),
                                            state_is_tuple=(self.rnn_cell == "LSTM"))
            stackb = tf.nn.rnn_cell.MultiRNNCell([cell] * (self.layers),
                                                state_is_tuple=(self.rnn_cell == "LSTM"))

            self.reset_state_stackf = stackf.zero_state(self.batch_size, dtype=tf.float32)
            
            self.reset_state_stackb = stackb.zero_state(self.batch_size, dtype=tf.float32)
            if self.insertLastState:
                if self.rnn_cell != "LSTM":
                    raise Exception("model type not supported for insertion of last state: {}".format(self.rnn_cell))
                    
                self.state_stackb = [[tf.placeholder(tf.float32,
                                                     [self.batch_size, self.hidden])]*2 for i in range(self.layers)]
                self.state_stackf = [[tf.placeholder(tf.float32,
                                                     [self.batch_size, self.hidden])]*2 for i in range(self.layers)]

                self.rnn_tuple_statef = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(self.state_stackf[idx][0], self.state_stackf[idx][1])
                 for idx in range(self.layers)]
                )

                self.rnn_tuple_stateb = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(self.state_stackb[idx][0], self.state_stackb[idx][1])
                 for idx in range(self.layers)]
                )       
                print("insertion of last state")
                #todo: try just initial states
                outputs, self.state_fw, self.state_bw  = tf.nn.bidirectional_rnn(stackf, stackb, x,
                                                                                 sequence_length=seq_len,
                                                                                 dtype=tf.float32,
                                                                                 initial_state_fw=self.rnn_tuple_statef,
                                                                                 initial_state_bw=self.rnn_tuple_stateb)
            else:
                print("no insertion of last state")
                outputs, self.state_fw, self.state_bw  = tf.nn.bidirectional_rnn(stackf, stackb, x,
                                                                                 sequence_length=seq_len,
                                                                                 dtype=tf.float32,
                                                                                 initial_state_fw=self.reset_state_stackf,
                                                                                 initial_state_bw=self.reset_state_stackb)
            #= states
            y_predict = tf.reshape(outputs, [-1, 2*self.hidden])
        return y_predict
        
        
        

  def loss(self, batch_x, batch_y=None):
    y_predict = self.inference(batch_x)
    self.optimizer, cost, ler, self.decoded = self.ctc_loss(y_predict,
                                                            batch_y, 
                                                            batch_x[1], 
                                                            self.num_classes, 
                                                            self.learning_rate)
    self.cost, self.ler = cost, ler
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('error', ler)

  def optimize(self):
    return self.optimizer

  def graphMaker(self):
        #self.graph = tf.Graph()
        with tf.Graph().as_default():
            self.train_batch = tf.placeholder_with_default(True,[])
            videoInputs_train, seq_len_train, targets_train  = self.inputs(self.input_path)
            videoInputs_test, seq_len_test, targets_test = self.inputs(self.input_path_test)
            targets_train.name = "train-sparse"
            targets_test.name = 'test-sparse'
            videoInputs, seq_len, targets = tf.cond(
                self.train_batch,
                lambda:[videoInputs_train, seq_len_train, targets_train],
                lambda:[videoInputs_test, seq_len_test, targets_test]
            )
            
            self.loss([videoInputs, seq_len], targets)  
            '''
            tf.cond(
                self.train_batch,
                lambda:tf.summary.scalar('loss-train', self.cost),
                lambda:tf.summary.scalar('loss-test', self.cost)
            )
            
            tf.cond(
                self.train_batch,
                lambda:tf.summary.scalar('loss-error', self.ler),
                lambda:tf.summary.scalar('loss-error', self.ler)
            )
            ''' 
            
            #optimizer = self.optimize()
            try:
                    self.summary_op = tf.summary.merge_all()
            except AttributeError:
                    self.summary_op = tf.merge_all_summaries()

            self.saver = tf.train.Saver()
            
            if self.gpu:
                ### start session
                config=tf.ConfigProto()
                # config.gpu_options.per_process_gpu_memory_fraction=0.98
                config.gpu_options.allocator_type="BFC"
                config.log_device_placement=True
                self.sess=tf.Session(config=config)
            else:
                self.sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            # Remove this if once Tensorflow 0.12 is standard.
            try:
                self.summary_writer_train = tf.summary.FileWriter(self.board_path+'/train', self.sess.graph)
                self.summary_writer_test = tf.summary.FileWriter(self.board_path+'/test', self.sess.graph)
            except AttributeError:
                self.summary_writer_test = tf.train.SummaryWriter(self.board_path+'/train', self.sess.graph)
                self.summary_writer_train = tf.train.SummaryWriter(self.board_path+'/test', self.sess.graph)

            self.global_step = 0
            try:
                    ckpt = tf.train.get_checkpoint_state(self.model_dir)
                    load_path = ckpt.model_checkpoint_path
                    self.saver.restore(self.sess, load_path)
            except:
                    print("no saved model to load. starting new session")
                    try:
                        init = tf.global_variables_initializer()
                        self.sess.run(init)
                    except AttributeError:
                        init = tf.initialize_all_variables()
                        self.sess.run(init)
            else:
                    print("loaded model: {}".format(load_path))
                    self.global_step = int(load_path.split('-')[-1]) 

            tf.train.start_queue_runners(sess=self.sess)
            

def run_training(args):    
    
        model = Model(args)
        '''
        sess = tf.Session()
        with sess.as_default():
            tf.train.start_queue_runners(sess=sess)
            f,g,h = model.read_and_decode_single_example(args.input_path)
            print(f.eval())
        return;
        '''
        model.graphMaker()
        state_b = None
        state_f = None
        totalE = 0.;
        totalL = 0.;
        # Start the training loop.
        start_time = time.time()
        for step in xrange(model.global_step, args.max_steps):
            if step == model.global_step and args.insertLastState:
                state_b, state_f = model.sess.run([model.reset_state_stackb, model.reset_state_stackf])
                
            feed = {}
            if args.insertLastState:
                for i in range(args.layers):
                    feed[model.state_stackb[i][0]] = state_b[i].c
                    feed[model.state_stackb[i][1]] = state_b[i].h
                    feed[model.state_stackf[i][0]] = state_f[i].c
                    feed[model.state_stackf[i][1]] = state_f[i].h


                loss, error, _, state_f, state_b = model.sess.run([model.cost,  
                                                                   model.ler, 
                                                                   model.optimizer,
                                                                   model.state_fw,
                                                                   model.state_bw],
                                                     feed_dict=feed)
            else:
                loss, error, _ = model.sess.run([model.cost,  
                                                                   model.ler, 
                                                                   model.optimizer],
                                                     feed_dict=feed)
            totalE += error
            totalL += loss
            
            if step % args.display_step == 0:
                
                loss, error, summary_str = model.sess.run([model.cost,
                                                           model.ler,
                                                           model.summary_op],feed_dict=feed)
                # Update the events file.
                model.summary_writer_train.add_summary(summary_str, step)
                model.summary_writer_train.flush()
                
                
                duration = time.time() - start_time
                feed[model.train_batch]=False
                loss, error, summary_str = model.sess.run([model.cost,
                                                           model.ler,
                                                           model.summary_op],feed_dict=feed)
                totalE /= args.display_step
                totalL /= args.display_step
                print ("step: {2}, test loss : {0}, test error: {1} ,".format(loss,error,step),
                "val loss : %f, train error = %.1f%%, time=%f ms" %(totalL, totalE, duration))
                
                # Update the events file.
                model.summary_writer_test.add_summary(summary_str, step)
                model.summary_writer_test.flush()
                
                start_time = time.time()
                totalE = 0.;
                totalL = 0.;

            # Save a checkpoint and evaluate the model periodically.
            if args.save_step!=1 and (step != model.global_step) \
            and ((step + 1) % args.save_step == 0 or (step + 1) == args.max_steps):
                checkpoint_file = os.path.join(args.model_dir, 'checkpoint')
                model.saver.save(model.sess, checkpoint_file, global_step=step)
                print('model saved!')
        
        #print(sess.run([indices]))
        
            
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
    parser.add_argument('--optimizer', type=str, default="ADAM", help='optimizer to use')
    parser.add_argument('--initializer', type=str, default="graves", help='initializer to use')
    parser.add_argument('--bias', type=float, default=0.1, help='initializer to use for bias')
    
    #iteration param
    parser.add_argument('--save_step', type=int, default=FLAGS.save_step, help='after how many iteration to save the step')
    parser.add_argument('--max_steps', type=int, default=FLAGS.max_steps, help='number of iterations')
    parser.add_argument('--display_step', type=int, default=FLAGS.display_step, help='number of iterations to display after')
    parser.add_argument('--batch_size', type=int, default=FLAGS.batch_size, help='the size of the batch')
    
    #files param
    parser.add_argument('--model_dir', type=str, default=FLAGS.model_dir, help='location to save model')
    parser.add_argument('--input_path', type=str, default=FLAGS.input_path, help='location to get data for training')
    parser.add_argument('--input_path_test', type=str, default=FLAGS.input_path_test, help='location to get data for testing')
    parser.add_argument('--board_path', type=str, default=FLAGS.board_path, help='location to save statistics')
    parser.add_argument('--filenameNr', type=str, default=1, help='if more than one use format(i) to make the files input')
    
    #static param
    parser.add_argument('--width', type=int, default=90, help='image width')
    parser.add_argument('--height', type=int, default=36, help='image height')
    parser.add_argument('--num_classes', type=int, default=FLAGS.num_classes, help='number of classes')
    
    #defaults
    parser.set_defaults(train=True)
    parser.set_defaults(insertLastState=False)
    parser.set_defaults(gpu=False)
    parser.set_defaults(distributed=False)
    args = parser.parse_args()
    
    run_training(args) if args.train else print("todo: show sample from the model")


if __name__ == '__main__':
    tf.app.run()
