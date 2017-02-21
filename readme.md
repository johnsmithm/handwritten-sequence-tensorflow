Tensorflow implementation of handwritten sequense of small letter recognition.

The handwritten dataset used is IAM.

In order to make TFRecord file use the function from util.py

In order to run the training: python trainer/run.py --input_path data/handwritten-test.tfrecords --input_path_test data/handwritten-test.tfrecords --model_dir models --board_path TFboard --filenameNr 1 --save_step 100  --batch_size 10 --max_steps 1000 --display_step 100
