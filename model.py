# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications Copyright 2018 Arman Cohan
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
from tensorflow.contrib.slim.python.slim import learning
import sys

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from util import load_embeddings
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import array_ops
from six.moves import xrange
from tensorflow.python import debug as tf_debug
FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab, num_gpus):
    self._hps = hps
    self._cur_gpu = 0
    self._vocab = vocab
    self._num_gpus = num_gpus
    if FLAGS.new_attention and FLAGS.hier:
      print('using linear attention mechanism for considering sections')
      from attention_decoder_new import attention_decoder
    else:
      print('using hierarchical attention mechanism for considering sections')
      from attention_decoder import attention_decoder
    self.attn_decoder = attention_decoder

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    if self._hps.hier:
      self._enc_batch_sections = tf.placeholder(tf.int32, [hps.batch_size, hps.num_sections, None], name='enc_batch_sections')
      self._doc_sec_lens = tf.placeholder(tf.int32, [hps.batch_size]) # length of doc in num sections
      self._batch_sections_len = tf.placeholder(tf.int32, [hps.batch_size, hps.num_sections])
      self._enc_section_padding_mask = tf.placeholder(tf.int32, [hps.batch_size, hps.num_sections, None], name='enc_section_padding_mask')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='padding_mask')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    if self._hps.hier:
      feed_dict[self._enc_batch_sections] = batch.batch_sections  # shape=[batch-size, num-sections, enc-seq-len]
      feed_dict[self._batch_sections_len] = batch.batch_sections_len # length of sections in the entire batch (num sections). [[400, 400, 400, 400],[...]]
      feed_dict[self._doc_sec_lens] = batch.batch_doc_sec_lens
      feed_dict[self._enc_section_padding_mask] = batch.enc_section_padding_mask
    else:
      feed_dict[self._enc_batch] = batch.enc_batch
      feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    feed_dict[self._enc_lens] = batch.enc_lens
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st


  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state
    into a single initial state for the decoder. This is needed
    because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of
        (batch_size, vsize) arrays. The words are in the order they appear in the
        vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of
        (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of
        (batch_size, extended_vsize) arrays. extended_vsize is the vocab + article OOV
    """

    with tf.variable_scope('final_distribution'):
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

      # Extend the vocabulary dist with zeros (for OOV words
      extended_vsize = self._vocab.size() + self._max_art_oovs 
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
      # list length max_dec_steps of shape (batch_size, extended_vsize)
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros])
                              for dist in vocab_dists]


      # Project the values in the attention distributions onto the appropriate
      # entries in the final distributions
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)

      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] 
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) 
      shape = [self._hps.batch_size, extended_vsize]
      # indices has shape [batch_size, extended_vsize, 2]
      # sample slice: [[[0, 701], ... ], [[1, 529], ...], [[2, 728], ...]]
      # scatter the distribution among corresponding batches and vocabulary index
      

      # list length max_dec_steps (batch_size, extended_vsize)
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape)
                              for copy_dist in attn_dists]

      # Add a very small epsilon
      def _add_epsilon(dist, epsilon=sys.float_info.epsilon):
        epsilon_mask = tf.ones_like(dist) * epsilon
        return dist + epsilon_mask
      # Add the vocab distributions and the copy distributions together to get
      # the final distributions
      final_dists = [vocab_dist + copy_dist
                     for (vocab_dist, copy_dist)
                     in zip(vocab_dists_extended, attn_dists_projected)]
      final_dists = [_add_epsilon(dist) for dist in final_dists]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    chkpt_dir = tf.train.latest_checkpoint(train_dir)
    print('chkpt_dir for embeddings: ', chkpt_dir)
    if chkpt_dir:
      config.model_checkpoint_path = chkpt_dir
    else:
      chkpt_dir = train_dir
    projector.visualize_embeddings(summary_writer, config)
    
  def _next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self._num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self._cur_gpu
    if self._num_gpus > 1:
      self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus-1)
    return dev
  
  def _get_gpu(self, gpu_id):
    if self._num_gpus <= 0 or gpu_id >= self._num_gpus:
      return ''
    return '/gpu:%d' % gpu_id

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary
    
    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)


      with tf.variable_scope('embedding'):
        if hps.pretrained_embeddings:
          word2vec = load_embeddings(hps.embeddings_path, self._vocab.word2id, hps.rand_unif_init_mag)
          self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim],
                                    dtype=tf.float32, initializer=tf.constant_initializer(word2vec))
          # self.assign_embedding = tf.assign(self.embedding, word2vec)
        else:
          self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim],
                                      dtype=tf.float32, initializer=self.trunc_norm_init)
        if hps.mode=="train": self._add_emb_vis(self.embedding) # add to tensorboard

        # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_enc_inputs = tf.nn.embedding_lookup(self.embedding, self._enc_batch)
        if self._hps.hier:
          enc_batch_sections = tf.unstack(self._enc_batch_sections, axis=1)
          sec_emb_enc_inputs = [tf.nn.embedding_lookup(self.embedding, section)
                                for section in enc_batch_sections]
        # list length max_dec_steps containing shape (batch_size, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(self.embedding, x)
                          for x in tf.unstack(self._dec_batch, axis=1)]


      # Hierarchical attention model
      if self._hps.hier:
        with tf.variable_scope('encoder'), tf.device(self._next_device()):
          sec_enc_outs = []
          states_fw = []
          states_bw = []
          states = []

          # level 1, encode words to sections
          with tf.variable_scope("word_level_encoder", reuse=tf.AUTO_REUSE) as scope:
            encoder_outputs_words = []
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            fw_st, bw_st = None, None
            if self._hps.use_do:  # DropOut
              cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1.0 - self._hps.do_prob)
              cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=1.0 - self._hps.do_prob)
            for i in range(self._hps.num_sections):
              encoder_tmp_output, (fw_st, bw_st) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs=sec_emb_enc_inputs[i], dtype=tf.float32,
                sequence_length=self._batch_sections_len[:,i], swap_memory=True, initial_state_bw=bw_st, initial_state_fw=fw_st)
              # concatenate the forwards and backwards states
              encoder_tmp_output = tf.concat(axis=2, values=encoder_tmp_output) #shape=[batch x seq_len x hidden_size]
                 
              encoder_outputs_words.append(encoder_tmp_output)
              # instead of concating the fw and bw states, we use a ff network
              combined_state = self._reduce_states(fw_st, bw_st)
              states.append(combined_state)
              scope.reuse_variables()
              
          # level 2, encode sections to doc
          encoder_outputs_words = tf.stack(encoder_outputs_words, axis=1)  # shape [batch x num_sections x seq_len x hidden_size]
          shapes = encoder_outputs_words.shape
          encoder_outputs_words = tf.reshape(encoder_outputs_words, (shapes[0].value, -1, shapes[-1].value))  #shape=[batch x (seq_len * num_sections) x hidden_size]

          doc_sections_h = tf.stack([s.h for s in states], axis=1)   # [batch x num_sections x hidden_size]
          doc_sections_c = tf.stack([s.c for s in states], axis=1)   # [batch x num_sections x hidden_size]

          with tf.variable_scope("section_level_encoder"):
            if FLAGS.section_level_encoder == 'RNN':
              cell_fw_1 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
              cell_bw_1 = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
              if self._hps.use_do:
                  cell_fw_1 = tf.contrib.rnn.DropoutWrapper(cell_fw_1, output_keep_prob=1.0 - self._hps.do_prob)
                  cell_bw_1 = tf.contrib.rnn.DropoutWrapper(cell_bw_1, output_keep_prob=1.0 - self._hps.do_prob)
              encoder_output_sections, (fw_st_2, bw_st_2) =\
                  tf.nn.bidirectional_dynamic_rnn(cell_fw_1, cell_bw_1, inputs=doc_sections_h, sequence_length=self._doc_sec_lens, dtype=tf.float32, swap_memory=True)
              encoder_output_sections = tf.concat(axis=2, values=encoder_output_sections)
              doc_sections_state = self._reduce_states(fw_st_2, bw_st_2)
            else:
              if FLAGS.section_level_encoder == 'AVG': # average section cells
                doc_sections_state_h = tf.reduce_mean(doc_sections_h, axis=1)
                doc_sections_state_c = tf.reduce_mean(doc_sections_c, axis=1)
              elif FLAGS.section_level_encoder == 'FF':  # use a feedforward network to combine section cells
                doc_sections_state_h = tf.reshape([doc_sections_h.shape[0].eval(), -1])
                doc_sections_state_h = tf.layers.dense(
                  inputs=doc_sections_state_h,
                  units=self._hps.hidden,
                  activation=tf.nn.relu)              
                doc_sections_state_c = tf.reshape([doc_sections_c.shape[0].eval(), -1])
                doc_sections_state_c = tf.layers.dense(
                  inputs=doc_sections_state_c,
                  units=self._hps.hidden,
                  activation=tf.nn.relu)
              else:
                raise AttributeError('FLAGS.section_level_encoder={} is not a valid option'.format(FLAGS.section_level_encoder))
              doc_sections_state = tf.contrib.rnn.LSTMStateTuple(doc_sections_state_c, doc_sections_state_h)
              encoder_output_sections = doc_sections_h         
      
      elif not self._hps.multi_layer_encoder:
        with tf.variable_scope('encoder'):
          with tf.variable_scope('word_level_encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) =\
              tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=emb_enc_inputs, dtype=tf.float32, sequence_length=self._enc_lens, swap_memory=True)
            # concatenate the forwards and backwards states
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
    
      # stack n layers of lstms for encoder
      elif self._hps.multi_layer_encoder:
        # TODO: check
        for layer_i in xrange(self._hps.enc_layers):
          with tf.variable_scope('encoder%d'%layer_i), tf.device(
              self._next_device()):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            if self._hps.use_do: # add dropout
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1.0 - self._hps.do_prob)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=1.0 - self._hps.do_prob)
            emb_enc_inputs, (fw_st, bw_st) =\
              tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=emb_enc_inputs, dtype=tf.float32, sequence_length=self._enc_lens, swap_memory=True)
            emb_enc_inputs = tf.concat(axis=2, values=emb_enc_inputs)
        encoder_outputs = emb_enc_inputs
      
      if self._hps.hier:
        self._enc_sec_states = encoder_output_sections
        self._enc_states = encoder_outputs_words 
      else:
        self._enc_states = encoder_outputs
        self._enc_sec_states = None
       
      # convert the encoder bidirectional hidden state to the decoder state
      # (unidirectional) by an MLP
      if self._hps.hier:
        self._dec_in_state = doc_sections_state
      else:
        with tf.variable_scope('encoder'):
          with tf.variable_scope('word_level_encoder'):
            self._dec_in_state = self._reduce_states(fw_st, bw_st) 
        
      # Add the decoder

      with tf.variable_scope('decoder'), tf.device(self._next_device()):
        cell = tf.contrib.rnn.LSTMCell(
          self._hps.hidden_dim,
          state_is_tuple=True,
        initializer=self.rand_unif_init)
    
        # We need to pass in the previous step's coverage vector each time
        prev_coverage = self.prev_coverage\
                         if hps.mode=="decode" and self._hps.coverage \
                         else None 
    
        
        if self._hps.hier:
          decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage, self.attn_dists_sec =\
            self.attn_decoder(emb_dec_inputs,
                              self._dec_in_state,
                              self._enc_states,
                              cell,
                              self._enc_sec_states,
                              num_words_section=self._batch_sections_len,
                              enc_padding_mask=self._enc_padding_mask,
                              enc_section_padding_mask=self._enc_section_padding_mask,
                              initial_state_attention=(self._hps.mode=="decode"),
                              pointer_gen=self._hps.pointer_gen,
                              use_coverage=self._hps.coverage,
                              prev_coverage=prev_coverage,
                              temperature=self._hps.temperature
                              )
                      
        else:
          decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage, _ =\
            self.attn_decoder(emb_dec_inputs,
                              self._dec_in_state,
                              self._enc_states,
                              cell,
                              encoder_section_states=None,
                              num_words_section=None,
                              enc_padding_mask=self._enc_padding_mask,
                              initial_state_attention=(self._hps.mode=="decode"),
                              pointer_gen=self._hps.pointer_gen,
                              use_coverage=self._hps.coverage,
                              prev_coverage=prev_coverage,
                              )      
        

      # Project decoder output to vocabulary
      with tf.variable_scope('output_projection'), tf.device(self._next_device()):
        if self._hps.output_weight_sharing:
          # share weights of embedding layer with projection
          # self.embedding is in shape [vsize, hps.emb_dim]
          w_proj = tf.get_variable('w_proj', [self._hps.emb_dim, self._hps.hidden_dim],
                              dtype=tf.float32, initializer=self.trunc_norm_init)
          w = tf.tanh(tf.transpose(tf.matmul(self.embedding, w_proj))) # shape = [vsize, hps.hidden_dim]
          
  #         w_t = tf.transpose(w)
          b = tf.get_variable('b', [vsize],
                              dtype=tf.float32, initializer=self.trunc_norm_init)
        else: 
          w = tf.get_variable('w', [self._hps.hidden_dim, vsize],
                              dtype=tf.float32, initializer=self.trunc_norm_init)
  #         w_t = tf.transpose(w)
          b = tf.get_variable('b', [vsize],
                              dtype=tf.float32, initializer=self.trunc_norm_init)
        # vocabulary score at each decoder step
        vocab_scores = []
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          vocab_scores.append(tf.nn.xw_plus_b(output, w, b)) # apply the linear layer

        # the final vocab distribution for each decoder time step
        # shape of each element is [batch_size, vsize]
        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] 

      
      # pointing / generating
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
#         log_dists = [tf.log(dist) for dist in final_dists]
      else:
#         log_dists = [tf.log(dist) for dist in vocab_dists]
        final_dists = vocab_dists
        

      # Calculate Losses:
      
      if self._hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'), tf.device(self._next_device()):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the gold target words
            # will be list length max_dec_steps containing shape (batch_size)
            loss_per_step = [] 
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              # The indices of the target words. shape (batch_size)
              targets = self._target_batch[:,dec_step] 
              indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
              # shape (batch_size). loss on this step for each batch
              gold_probs = tf.gather_nd(dist, indices)
              losses = -tf.log(gold_probs)
              loss_per_step.append(losses)

            # Apply dec_padding_mask mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
            

          else: # baseline model
            # this applies softmax internally
            self._loss = tf.contrib.seq2seq.sequence_loss(
              tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          if self._hps.coverage:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            self._total_loss = self._loss + self._hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)
            
        # ---------------------------/


    if self._hps.mode == "decode":
        assert len(final_dists) == 1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
        final_dists = final_dists[0]
        topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
        self._topk_log_probs = tf.log(topk_probs)

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    self._lr_rate = tf.maximum(
      self._hps.min_lr,  # min_lr_rate.
      tf.train.exponential_decay(self._hps.lr, self.global_step, 30000, 0.98))
    
    
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    # Clip the gradients
    with tf.device(self._get_gpu(self._num_gpus-1)):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    if self._hps.optimizer == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
        self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)

    elif self._hps.optimizer == 'adam':    
      # Adam
      optimizer = tf.train.AdamOptimizer()
    
    elif self._hps.optimizer == 'sgd':
      # SGD
      optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
      tf.summary.scalar('learning rate', self._lr_rate)
    
    else:
      raise Exception('Invalid optimizer: ', self._hps.optimizer)

    with tf.device(self._get_gpu(self._num_gpus-1)):
      self._train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)
    
    print('#'*78,'\nprinting model variables:')
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape().as_list()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        print('{:}: shape={:}, variable_parameters={:}'.format(
          variable.name, shape, variable_parameters))
        total_parameters += variable_parameters
    print('total model parameters: {:}'.format(total_parameters))
    print('#'*78)

  def run_train_step(self, sess, batch):
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    if FLAGS.debug:
      print('entering debug mode\n\n\n\n\n\n\n\n\n')
      sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=FLAGS.dump_root, ui_type=FLAGS.ui_type)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
#     try:    
    res = sess.run(to_return, feed_dict)

    if not np.isfinite(res['loss']):
      print('loss is nan!!!!!')
      raise Exception("Loss is not finite. Stopping.")
    
#     except tf.errors.InvalidArgumentError:
#       import pdb; pdb.set_trace()
    return res

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    res = sess.run(to_return, feed_dict)
    return res

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) 
    (enc_states, dec_in_state, global_step) = sess.run(
      [self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists
    }
    
    if self._hps.hier:
      feed[self._enc_batch_sections] = batch.batch_sections  # shape=[batch-size, num-sections, enc-seq-len]
      feed[self._batch_sections_len] = batch.batch_sections_len
      feed[self._doc_sec_lens] = batch.batch_doc_sec_lens
      feed[self._enc_section_padding_mask] = batch.enc_section_padding_mask
      feed[self._enc_lens] = batch.enc_lens
      to_return['attn_dists_sec'] = self.attn_dists_sec

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()
    
    if 'attn_dists_sec' in results:
      if len(results['attn_dists_sec']) > 0:
        attn_dists_sec = results['attn_dists_sec'][0].tolist()
      else: attn_dists_sec = None
    else:
      attn_dists_sec = None

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage, attn_dists_sec


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
