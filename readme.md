Tensorflow implementation of handwritten sequense of small letter recognition.

The handwritten dataset used is IAM.

In order to make TFRecord file use the function from util.py

In order to run the training: python trainer/run.py --input_path data/handwritten-test.tfrecords --input_path_test data/handwritten-test.tfrecords --model_dir models --board_path TFboard --filenameNr 1 --save_step 500  --batch_size 10 --max_steps 1000 --display_step 100


Google cloud ML:

gcloud beta ml jobs submit training handwrittenv16 \
  --package-path=trainer \
  --module-name=trainer.run \
  --staging-bucket=gs://my-first-bucket-mosnoi/ \
  --region=us-central1 \
  --scale-tier=BASIC_GPU
  -- /
  --input_path gs://my-first-bucket-mosnoi/handwritten/m1/tf-data/handwritten-test-{}.tfrecords \
  --input_path_test gs://my-first-bucket-mosnoi/handwritten/m1/tf-data/handwritten-test-55.tfrecords \
  --board_path gs://my-first-bucket-mosnoi/handwritten/m1/TFboard \
  --model_dir gs://my-first-bucket-mosnoi/handwritten/m1/models \
  --filenameNr 50 \
  --save_step 15000 \
  --display_step 100 \
  --max_steps 40000 \
  --save_step 100 \
  --batch_size 20 \
  --learning_rate 0.0001 \
  --keep_prob 0.8 \
  --layers 3 \
  --hidden 200 \
  --rnn_cell LSTM \
  --optimizer ADAM \ 
  --initializer  graves \
  --bias 0.1 \
  --gpu 
  
  //--optimizer RMSP --momentum 0.02 --decay 
  // python run.py --insertLastState
  
  