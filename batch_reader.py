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
from itertools import chain
import sys
"""This file contains code to process data into batches"""

from six.moves import queue as Queue
from six.moves import xrange
import six
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data

# To represent list of sections as string and retrieve it back
SECTION_SEPARATOR = ' <SCTN/> '

# to represent separator as string, end of item (ei)
LIST_SEPARATOR = ' <EI/> '


def _string_to_list(s, dtype='str'):
    """ converts string to list
    Args:
      s: input
      dtype: specifies the type of elements in the list
        can be one of `str` or `int`
    """
    if dtype == 'str':
        return s.split(LIST_SEPARATOR)
    elif dtype == 'int':
        return [int(e) for e in s.split(LIST_SEPARATOR) if e]


def _string_to_nested_list(s):
    return [e.split(LIST_SEPARATOR)
            for e in s.split(SECTION_SEPARATOR)]

# TEMP REMOVED, May be not needed
# def _count_words(s):
#   """ Count words in a list of strings """
#   return sum([len(e.split(' ')) for e in s])
  

def _section_to_ids(section, vocab, max_len, pad_id):
  """ Converts words in a section (list of strings) to ids and pad if necessary """
  section_text = ' '.join(section)
  section_words = section_text.split()
  sec_len = len(section_words)
  if sec_len > max_len:
    section_words = section_words[:max_len]
  word_ids = [vocab.word2id(w) for w in section_words]
  
  
def _flatten(lst):
  """ flatten a nested list (list of lists) """
  return list(chain.from_iterable(lst))

def _get_section_words(sec, max_len=None, pad_id=data.PAD_TOKEN, pad=True):
  """ given a section (list of sentences), returns a single list of words in that section """
  words = ' '.join(sec).split()
  if max_len is None:
    max_len = len(words)
  if pad:
    while len(words) < max_len:
      words += [pad_id]
  return words[:max_len]

def _pad_words(words, max_len=None, pad_id=data.PAD_TOKEN):
  """ given a section (list of sentences), returns a single list of words in that section """
  if max_len is None:
    max_len = len(words)
  while len(words) < max_len:
    words += [pad_id]
  return words[:max_len]
    



class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, article, abstract_sentences, article_id, sections, section_names, labels, vocab, hps):
        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        Args:
          article: source text; a list of strings. each token is separated by a single space.
          abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
          article_id: string
          sections: list of list of strings
          section_names: list of strings
          labels: list of strings, for extractive summarization training (TODO Later)
          vocab: Vocabulary object
          hps: hyperparameters
        """
        self.hps = hps
        self.discard = False
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)


        # clean section information
        # clean sections after conclusions
        if hps.hier:
          end_loc = len(section_names)
          beg_loc = 0
          for i,s in enumerate(section_names):
            if 'conclu' in s.lower():
              end_loc = i + 1
            if 'intro' in s.lower() and beg_loc == 0:
              beg_loc = i
              
          if beg_loc < len(section_names) - end_loc:
            sections = sections[beg_loc:end_loc]
          try:
            intro_last = sections[beg_loc][-2:] # last two sentences in the intro
          except IndexError:
#             print('article_id: {}, len(sections): {}, section_names: {}'.format(article_id, len(sections), section_names))
            self.discard = True
            return
#           intro_first = []
          i = 0
#           intro_last_len = _count_words(intro_last)
#           intro_len = intro_last_len
#           while(intro_len < hps.max_intro_len):
#             intro_first.append(sections[beg_loc][i])
#             intro_len = _count_words(intro_first) + intro_last_len
#             i += 1
          
          if not hps.split_intro:
          
            max_sents = hps.max_intro_sents - 2 # exclude the last two sents
            intro_first = sections[beg_loc][:max_sents]
            intro_last_words = _get_section_words(intro_last, pad=False)
            intro_last_len = len(intro_last_words) # flatten list of sents, get the string inside, count words
            
            discard_last = False
            if intro_last_len > hps.max_intro_len:
              discard_last = True
            len_limit = hps.max_intro_len - intro_last_len if not discard_last else hps.max_intro_len
            # truncate the intro by len_limit (we consider last 2 sentences from the intro to be there always)
            # Flatten list of lists, get the first element (string), get words, get first n words, return a striing, make it a list, extend it with intro_last
            intro_words = _get_section_words(intro_first, len_limit, pad=False)
            
            try:
              if intro_words[-1] != '.':
                intro_words = intro_words[:-1] + ['.']
                if not discard_last:
                  intro_words += intro_last_words
                intro_words = _pad_words(intro_words, hps.max_intro_len)
            except IndexError:
              print('No first section, Example discarded: ', article_id)
              self.discard = True
          
          else:    
            intro_first = sections[beg_loc][:hps.max_intro_sents]
            intro_words = _get_section_words(intro_first, hps.max_intro_len, pad=True)

          try:
            conclusion_words = _get_section_words(sections[end_loc - beg_loc - 1][:hps.max_conclusion_sents], hps.max_conclusion_len)
          except:
            import pdb; pdb.set_trace()
            print("ERROR, pause and check")
            print('end_loc:', end_loc)
            print('section_names:', section_names)
            print('num_sections: {}'.format(len(sections)))
            print('len_sections_sents:', [len(e) for e in sections])
            
#           if not hps.intro_split:
          article_sections = [_get_section_words(s[:hps.max_section_sents], hps.max_section_len)
                              for s in sections[1:-1][:hps.num_sections - 2]]
#           else:
#             tmp_sections = []
#             remaining_sec = sections[1:-1]
#             if len(remaining_sec) > hps.num_sections - 2:
#               for i in range(hps.num_sections - 2):
#                 tmp_sections.append(remaining_sec[i])
#               last_sec = []
#               while(i < len(remaining_sec)):
#                 last_sec.extend(remaining_sec[i])
#                 i += 1
#               tmp_sections.append(last_sec)
#               remaining_sec = tmp_sections
#   
#             article_sections = [_get_section_words(s, hps.max_section_len)
#                                 for s in remaining_sec]
          
          sections = [intro_words] + article_sections + [conclusion_words]
          sec_len = len(sections)
          self.sec_len = sec_len
          self.num_words_section = [hps.max_section_len for e in sections] 
          self.num_words_section_nopad = [len(e) for e in sections]
          # TODO: Assumption is that sections is a list of list (sections, sentences), check if assumption is true
          # TODO: Assumtpion is that number of sections is greater than 2, check if assumption is true
          
#           pad_id = vocab.word2id(data.PAD_TOKEN)
          
          
            
        

        article_text = ' '.join(article)
        # Process the article
        article_words = article_text.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        # store the length after truncation but before padding
        self.enc_len = len(article_words)
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words]
        
        if hps.hier:
          self.enc_sections = []
          
          for sec in sections:
            self.enc_sections.append([vocab.word2id(w) for w in sec])
          self.enc_sec_len = [len(e) for e in self.enc_sections]
#           self.enc_sec_len = sec_len # TODO: Check

        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        # list of word ids; OOVs are represented by the id for UNK token
        abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if hps.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are
            # represented by their temporary OOV id; also store the in-article
            # OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(
                article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are
            # represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(
                abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV
            # ids, the target now includes words that are in the article but
            # not in the abstract, so represented as OOV
            _, self.target = self.get_dec_inp_targ_seqs(
                abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

        self.article_id = article_id
        self.sections = sections
        self.section_names = section_names
        self.labels = labels

        # Store the original strings
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

    def pad_section_input(self, max_sec_len, max_secs, pad_id=None):
        """pad encoder sections with pad_id. if the number of sections is less than max_secs, add another section with all pads"""
#         print("Before padding -> len: {}, element_len: {}, element_type: {}".format(len(self.enc_sections), len(self.enc_sections[0]), type(self.enc_sections[0][0])))
        for i, sec in enumerate(self.enc_sections):
          while len(sec) < max_sec_len:
            self.enc_sections[i].append(pad_id)
        while len(self.enc_sections) < max_secs:
          pads = [pad_id for _ in range(max_sec_len)]
          self.enc_sections.append(pads)
          self.num_words_section.append(len(pads))
#           self.num_words_section.append(0)
          
#         print("After padding -> len: {}, element_len: {}, element_type: {}\n\n".format(len(self.enc_sections), len(self.enc_sections[0]), type(self.enc_sections[0][0])))
#         if self.hps.pointer_gen:
#             while len(self.enc_input_extend_vocab) < max_len:
#                 self.enc_input_extend_vocab.append(pad_id)



class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
           example_list: List of Example objects
           hps: hyperparameters
           vocab: Vocabulary object
        """
        self._hps = hps
        self.pad_id = vocab.word2id(
            data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.sec_pad_id = vocab.word2id(data.SEC_PAD_TOKEN)
        # initialize the input to the encoder
        self.init_encoder_seq(example_list, hps)
        # initialize the input and targets for the decoder
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
            self.enc_batch:
              numpy array of shape (batch_size, <=max_enc_steps) containing integer ids
              (all OOVs represented by UNK id), padded to length of longest sequence in the batch
            self.enc_lens:
              numpy array of shape (batch_size) containing integers.
              The (truncated) length of each encoder input sequence (pre-padding).

          If hps.pointer_gen, additionally initializes the following:
            self.max_art_oovs:
              maximum number of in-article OOVs in the batch
            self.art_oovs:
              list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
              Same as self.enc_batch, but in-article OOVs are represented by
              their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this
        # batch
        if hps.hier:
          max_enc_seq_len = hps.max_section_len * hps.num_sections
        else:
          max_enc_seq_len = max([ex.enc_len for ex in example_list])
          

        # Pad the encoder input sequences up to the length of the longest
        # sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension)
        # for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros(
            (hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
              self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs)
                                     for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros(
                (hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i,
                                            :] = ex.enc_input_extend_vocab[:]
                                            
        if self._hps.hier:
          # TODO: see if you can uncomment it. Doesn't work because of unstack in the model
#           max_num_sections = max([ex.sec_len for ex in example_list])
          max_num_sections = self._hps.num_sections
          max_num_sections_nopad = max([ex.sec_len for ex in example_list])
          for ex in example_list:
#             ex.pad_section_input(max_num_sections, self.sec_pad_id)
            ex.pad_section_input(self._hps.max_section_len, max_num_sections, self.pad_id)
          self.batch_doc_sec_lens = [max_num_sections for _ in example_list]
          self.batch_sections = np.zeros((hps.batch_size, max_num_sections, self._hps.max_section_len), dtype=np.int32)
          self.batch_sections_len = np.zeros((hps.batch_size, max_num_sections), dtype=np.int32)
          self.batch_sections_len_nopad = np.zeros((hps.batch_size, max_num_sections_nopad), dtype=np.int32)
          self.enc_section_padding_mask = np.zeros((hps.batch_size, max_num_sections, self._hps.max_section_len), dtype=np.float32)
          for i, ex in enumerate(example_list):
            j = 0
            while(j < len(ex.enc_sections)):
              self.batch_sections[i, j, :] = ex.enc_sections[j][:self._hps.max_section_len]
              if j < len(ex.enc_sec_len):
                for k in range(ex.enc_sec_len[j]):
                  self.enc_section_padding_mask[i][j][k] = 1
              j += 1
            self.batch_sections_len[i, :] = ex.num_words_section[:]
            try:
              self.batch_sections_len_nopad[i, :] = ex.num_words_section_nopad[:]
            except ValueError: # in cases that we want to assign a length 3 list to length 4
              for k in range(len(ex.num_words_section_nopad)):
                self.batch_sections_len_nopad[i, k] = ex.num_words_section_nopad[k]
              
            

    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
            self.dec_batch:
              numpy array of shape (batch_size, max_dec_steps),
              containing integer ids as input for the decoder, padded to max_dec_steps length.
            self.target_batch:
              numpy array of shape (batch_size, max_dec_steps),
              containing integer ids for the target sequence, padded to max_dec_steps length.
            self.dec_padding_mask:
              numpy array of shape (batch_size, max_dec_steps),
              containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch;
              0s correspond to padding.
            """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch
        # (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding.
        # However I believe this is possible, or will soon be possible, with Tensorflow 1.0,
        # in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros(
            (hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros(
            (hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros(
            (hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in xrange(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [
            ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [
            ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [
            ex.original_abstract_sents for ex in example_list]  # list of list of lists
        self.article_ids = [ex.article_id for ex in example_list]


class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass,
                 article_id_key,
                 article_key, abstract_key,
                 labels_key,
                 section_names_key,
                 sections_key):
        """Initialize the batcher. Start threads that process the data into batches.

        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          hps: hyperparameters
          single_pass: If True, run through the dataset exactly once
          (useful for when you want to run evaluation on the dev or test set).
          Otherwise generate random batches indefinitely (useful for training).
          article_id_key: article id key in tf.Example
          article_key: article feature key in tf.Example.
          abstract_key: abstract feature key in tf.Example.
          labels_key: labels feature key in tf.Example,
          section_names_key: section names key in tf.Example,
          sections_key: sections key in tf.Example,
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of
        # Examples waiting to be batched
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(
            self.BATCH_QUEUE_MAX * self._hps.batch_size)

        self._article_id_key = article_id_key
        self._article_key = article_key
        self._abstract_key = abstract_key
        self._labels_key = labels_key
        self._section_names_key = section_names_key
        self._sections_key = sections_key

        # Different settings depending on whether we're in single_pass mode or
        # not
        if single_pass:
            # just one thread, so we read through the dataset just once
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1  # just one thread to batch examples
            # only load one batch's worth of examples before bucketing; this
            # essentially means no bucketing
            self._bucketing_cache_size = 1
            # this will tell us when we're finished reading the dataset
            self._finished_reading = False
        else:
            self._num_example_q_threads = 16  # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            # how many batches-worth of examples to load into cache before
            # bucketing
            self._bucketing_cache_size = 100

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in xrange(self._num_example_q_threads):
            self._example_q_threads.append(
                Thread(target=self._fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in xrange(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self._fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if
        # they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        """Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example 
        repeated beam_size-many times; this is necessary for beam search.

        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch.'
                ' Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(),
                self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info(
                    "Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def _fill_example_queue(self):
        """Reads data from file and processes into Examples which are then placed into the example queue."""

        input_gen = self.text_generator(
            data.example_generator(self._data_path, self._single_pass))
        cnt = 0
        fail = 0
        while True:
            try:
                # read the next example from file. article and abstract are
                # both strings.
                (article_id, article_text, abstract_sents, labels,
                 section_names, sections) = six.next(input_gen)
            except StopIteration:  # if there are no more examples:
                tf.logging.info(
                    "The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        "single_pass mode is off but the example generator is out of data; error.")

            # Use the <s> and </s> tags in abstract to get a list of sentences.
#       abstract_sentences = [sent.strip() for sent in data.abstract2sents(''.join(abstract_sents))]
            abstract_sentences = [e.replace(data.SENTENCE_START, '').replace(data.SENTENCE_END, '').strip()
                                  for e in abstract_sents]

            
            # at least 2 sections, some articles do not have sections
            if "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ __ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _" in article_text:
              continue 
            
            if len(sections) <= 1:
              continue
            
            if not sections or len(sections) == 0:
              continue
            # do not process that are too long
            if len(article_text) > self._hps.max_article_sents:
              continue
              
            # Do not process documents with unusually long or short abstracts
            abst_len = len(' '.join(abstract_sentences).split())
            if abst_len > self._hps.max_abstract_len or\
                    abst_len < self._hps.min_abstract_len:
                continue
            
            # Process into an Example.
            example = Example(article_text, abstract_sentences, article_id, sections, section_names, labels,
                              self._vocab, self._hps)
            # place the Example in the example queue.
            if example.discard:
              fail += 1
            cnt += 1
            if example is not None and not example.discard:
              self._example_queue.put(example)
            if cnt % 100 == 0:
              print('total in queue: {} of {}'.format(cnt - fail, cnt))

    def _fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length,
        processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            if self._hps.mode != 'decode':
                # Get bucketing_cache_size-many batches of Examples into a
                # list, then sort
                inputs = []
                for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                # sort by length of encoder sequence
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)

                # Group the sorted Examples into batches, optionally shuffle
                # the batches, and place in the batch queue.
                batches = []
                for i in xrange(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:  # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in xrange(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error(
                        'Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self._fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error(
                        'Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self._fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_gen):
        """Generates article and abstract text from tf.Example."""
        while True:
            e = six.next(example_gen)
            try:
                article_id = self._get_example_feature(e, self._article_id_key)
                article_text = self._get_example_feature(e, self._article_key)
                abstract_text = self._get_example_feature(
                    e, self._abstract_key)
                if not self._hps.pubmed:
                  labels = self._get_example_feature(e, self._labels_key)
                section_names = self._get_example_feature(
                    e, self._section_names_key)
                sections = self._get_example_feature(e, self._sections_key)

                # convert to list
                article_text = _string_to_list(article_text)
                abstract_text = _string_to_list(abstract_text)
                if not self._hps.pubmed:
                  labels = _string_to_list(labels, dtype='int')
                else:
                  labels = None
                section_names = _string_to_list(section_names)
                sections = _string_to_nested_list(sections)  # list of lists
            except ValueError:
                tf.logging.error(
                    'Failed to get article or abstract from example')
                continue

            yield (article_id, article_text, abstract_text, labels, section_names, sections)

    def _get_example_feature(self, ex, key):
        """Extract text for a feature from td.Example.

        Args:
          ex: tf.Example.
          key: key of the feature to be extracted.
        Returns:
          feature: a feature text extracted.
        """
        return ex.features.feature[key].bytes_list.value[0].decode(
            'utf-8', 'ignore')
