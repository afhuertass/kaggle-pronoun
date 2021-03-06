# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt

from dnc import dnc
from dnc import repeat_copy
from dnc import pronun_data as pronoun_data
import pandas as pd 
import numpy as np
from bpe import Encoder 

from sklearn.metrics import log_loss
FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 128 , "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 128, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 32 , "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 20 , "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 1000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 10,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 100,
                        "Checkpointing step interval.")


def get_dnc( output_size = 3 ):

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value
  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)

  return dnc_core , initial_state 

def run_model2( dnc_cell , input_sequence  ):
    initial_state = dnc_cell.initial_state(FLAGS.batch_size)
    output_sequence , _ = tf.nn.dynamic_rnn(
      cell=dnc_cell,
      inputs=input_sequence,
      time_major=False,
      initial_state=initial_state

      )

    return output_sequence 
def highlight(  x ):
	    
	    # x is a row
	offset_pronoun = x["Pronoun-offset"] 
    
	pronoun = x["Pronoun"]
	len_pronoun = len( pronoun )
    #print( len_pronoun )
	text = x["Text"]
    
	new_text = text[: offset_pronoun] + "*" + pronoun +"*" + text[  offset_pronoun+len_pronoun:  ]
    
	return new_text

def name_replace( s, r1, r2):
	s = str(s).replace(r1,r2)
	return s

def preparedata( df ):

		df["A-coref"] = df["A-coref"].astype(int)
		df["B-coref"] = df["B-coref"].astype(int )
		df.loc[: , "N"] = 1 - (df["A-coref"] + df["B-coref"])

		df["A"] = df["A"].apply( lambda x: x.lower()  )
		df["B"] = df["B"].apply( lambda x: x.lower() )


		df["Pronoun"] = df["Pronoun"].apply( lambda x: x.lower() )
		df["Text"] = df["Text"].apply( lambda x : x.lower() )

		# removing some things
		df["Text"] = df.apply(lambda x: x["Text"].replace("*" , "x")  , axis = 1   )

		#

		df["Text"] = df.apply(lambda x: highlight(x), axis = 1)

		df["Text"] = df.apply(lambda x: name_replace( x["Text"] , x["A"] , "subjectone" ) , axis = 1   )
		df["Text"] = df.apply(lambda x: name_replace( x["Text"] , x["B"] , "subjecttwo" ) , axis = 1   )

		return df 


def fit_encoder( df ):

	df = preparedata( df )
	encoder = Encoder( 200 , pct_bpe = 0.88 )
	encoder.fit( df["Text"].values )


	return encoder

def train(num_training_iterations, report_interval , batch_size ):
  """Trains the DNC and periodically reports the loss."""
  df_train = pd.read_csv("./data/gap-test.tsv" , sep = "\t")
  df_test = pd.read_csv( "./data/gap-development.tsv" , sep = "\t") #[:100]

  encoder = fit_encoder( df_train )
  dataset = pronoun_data.Data( df_train , batch_size = batch_size , encoder = encoder  )
  dataset_test = pronoun_data.Data( df_test , batch_size = batch_size , encoder = encoder )


  dataset( shuffle = True )
  dataset_test( repeat = 1 )
  print( "fokiu iueputa")
  print( dataset.dataset1.output_shapes )
  print( dataset_test.dataset1.output_shapes )

  iterator = tf.data.Iterator.from_structure( dataset.dataset1.output_types , dataset.dataset1.output_shapes )
  train_init_op = iterator.make_initializer( dataset.dataset1  )
  test_init_op = iterator.make_initializer(  dataset_test.dataset1 )

  features, labels = iterator.get_next()

  #dataset_tensors = dataset(  )
  #dataset_tensors_test = dataset_test(  repeat = 1 )

  #inputs = tf.zeros( shape = [128 , 1 , 900 ])
  dnc_cell , _  = get_dnc()

  # train ops 
  output_logits = run_model2( dnc_cell ,  features )
  output_logits = output_logits[ : , -1 , :]
  output_sigmoid = tf.nn.sigmoid( output_logits )

  ## test operations

  #output_test = run_model2( dnc_cell , dataset_tensors_test[0] )
  #output_test_logits = output_test[: , -1 , : ]
  #output_test_sigmoid = tf.nn.sigmoid( output_test_logits )


  train_loss = dataset.loss( labels , output_sigmoid  )
  print( train_loss.shape )
  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  optimizer = tf.train.RMSPropOptimizer(
      FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

  saver = tf.train.Saver()

  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  # Train.
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    writer = tf.summary.FileWriter("output", sess.graph)
    start_iteration = sess.run(global_step)
    total_loss = 0
    sess.run( train_init_op )
    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss = sess.run([train_step, train_loss])
      total_loss += loss
      #print( loss )
      if (train_iteration + 1) % report_interval == 0:
        print( "train iteration:" , train_iteration )
        print( total_loss/ report_interval )
        #tf.logging.info("%d: Avg training loss %f.\n%s",
         #               train_iteration, total_loss / report_interval,
          #              )
        total_loss = 0
    writer.close()
    print( "Calculationg data test")
    preds = np.zeros( ( dataset_test.num_samples , dataset_test.output_size ) )
    actuals = np.zeros( (dataset_test.num_samples ,  dataset_test.output_size ))
    start = 0 
    end = start + batch_size
    sess.run( test_init_op )
    while True:
        try:
            pred , actual = sess.run( [output_sigmoid , labels ])
            delta = pred.shape[0]
            #print( delta )
            #print( pred.shape )
            #print( actual.shape )
            print( start )
            preds[ start: start+delta , : ] = pred
            actuals[ start:start+delta , :  ] = actual 
            start += delta
            
        except tf.errors.OutOfRangeError:
            #loss_t = sess.run( [ ]
            error = log_loss( actuals , preds )
            print( "Loss on test:" , error )
            break 




def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  Nepochs =  4 

  train( 4*100 , FLAGS.report_interval , FLAGS.batch_size )


if __name__ == "__main__":
  tf.app.run()
