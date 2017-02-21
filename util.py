from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def makeTFRecordFile(s):
        """Create TFRecord files
        @param s:   s[i][0] - [36,90] array, the image, float32,
                    s[i][2] - the length of the image with handwritting data, <90 , int32
                    s[i][1][0] - [?] array of int32, reprezenting the text in the image, vocabulary length = 29 (small letters)
        save the file: trainer/data/handwritten-test-0.tfrecords
        """
        # iterate over each example
        # wrap with tqdm for a progress bar

        writer = tf.python_io.TFRecordWriter("trainer/data/handwritten-test-{}.tfrecords".format(0))
        for ii in range(len(s[1])):
            # construct the Example proto boject
            example = tf.train.Example(
                # Example contains a Features proto object
                features=tf.train.Features(
                  # Features contains a map of string to Feature proto objects
                  feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'seq_len': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[s[2][ii]])),
                    'target': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=s[1][ii][0].astype("int64"))),
                    'imageInput': tf.train.Feature(
                        float_list=tf.train.FloatList(value=(s[0][ii]-0.5).reshape(-1).astype("float"))),
            }))
            # use the proto object to serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized object to disk
            writer.write(serialized)
        writer.close()
        #videoInputs, seq_len , target