##Tensorflow implementation of handwritten sequense of small letter recognition.

The handwritten dataset used is IAM.

In order to make TFRecord file use the function from util.py

In order to run the training: 
```shell
python trainer/run.py --input_path data/handwritten-test.tfrecords --input_path_test data/handwritten-test.tfrecords --model_dir models --board_path TFboard --filenameNr 1 --save_step 500  --batch_size 10 --max_steps 1000 --display_step 100
```

In order to get a sample from trained model:
```shell
python run.py --sample --shuffle_batch --batch_size 1 --input_path data/handwritten-test.tfrecords --input_path_test data/handwritten-test.tfrecords --model_dir models --board_path TFboard --filenameNr 1
```

#params:
  * --rnn_cell \[LSTM,GRU,LSTMGRID2,GRUGRID2,BasicLSTM,LSTMGRID\]
  * --optimizer \[ADAM,RMSP\]
  * --initializer  \[graves\]
  * --sample {make a sample}
  * --shuffle_batch {shuffle the batch}
  * --insertLastState {insert last state at next training step}
  * --ctc_decoder \[greedy,beam_search\]

##Google cloud ML:
```shell
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
  ```
  
  ```shell
  //--optimizer RMSP --momentum 0.9 --decay 0.95
  // python run.py --layers 1 --hidden 20 --rnn_cell GRUGRID2 --optimizer RMSP --insertLastState
  // python run.py  --shuffle_batch  --sample --batch_size 1 
  ```
  
