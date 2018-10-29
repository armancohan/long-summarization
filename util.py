# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import traceback

"""This file contains some utility functions"""

import tensorflow as tf
from tensorflow.contrib import learn
import time
import os
import numpy as np
FLAGS = tf.app.flags.FLAGS


def load_embeddings(fpath, word_to_id, emd_init_var=0.25):
  """ loads pretrained embeddings into a dictionary 
  Args:
    fpath: file path to the text file of word embedddings
    word_to_id: mapping of word to ids from Vocab
    emd_init_var: initialization variance of embedding vectors
  returns:
    dictionary
  """
  f_iter = open(fpath)
  num_words, vector_size = list(map( int,next(f_iter).strip().split(' ')))
  np.random.seed(123)
  embd = np.random.uniform(-emd_init_var,emd_init_var,(len(word_to_id), vector_size))
  i = 0
  print('loading word embeddings')
  for line in f_iter:
    i += 1
    if i % 1000 == 0:
      print('{}K lines processed'.format(i), '\r')
    if i == 1:
      continue
    row = line.strip().split(' ')
    word = row[0]
    vec = list(map(float, row[1:]))
    embd[word_to_id[word]] = np.array(vec)
  print('embeddings loaded, vocab size: {}'.format(num_words))
  f_iter.close()
  return embd

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(saver, sess, ckpt_dir="train", latest_filename=None):
  """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    ran_once = False
    try:
      if latest_filename is None:
        latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
      if not ran_once:
        ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
        ran_once = True
      ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
      time.sleep(10)