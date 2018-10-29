import struct
import six
import glob
import random

import tensorflow as tf
from pathlib import Path
import os
import collections
import re
import pathlib

# To represent list of sections as string and retrieve it back
SECTION_SEPARATOR = ' <SCTN/> '
# to represent separator as string, end of item (ei)
LIST_SEPARATOR = ' <EI/> '

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
    if ' <SENT/> ' in s:
      return s.split(' <SENT/> ')
    if dtype == 'str':
        return s.split(LIST_SEPARATOR)
    elif dtype == 'int':
        return [int(e) for e in s.split(LIST_SEPARATOR) if e]


def _string_to_nested_list(s):
    return [e.split(LIST_SEPARATOR)
            for e in s.split(SECTION_SEPARATOR)]


def _nested_list_to_string(sections):
  res = []
  for sect in sections:
    sect_sents = LIST_SEPARATOR.join(sect)
    res.append(sect_sents)
  return SECTION_SEPARATOR.join(res)


class BinReader:

    def __init__(self, data_path, single_pass=True, return_type='dict', pubmed=False):
        self.single_pass = single_pass
        self.data_path = data_path
        self.return_type = return_type
        self._pubmed = pubmed

    def example_generator(self):
        """Generates tf.Examples from data files.

          Binary data format: <length><blob>. <length> represents the byte size
          of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
          the tokenized article text and summary.

        Args:
          data_path:
            Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
          single_pass:
            Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.
          return_type:
            can be 'dict' or 'tuple'

        Yields:
          Deserialized tf.Example.
        """
        epoch = 0
        while True:
            num_files = 0
            filelist = glob.glob(self.data_path)  # get the list of datafiles
            assert filelist, ('Error: Empty filelist at %s' %
                              self.data_path)  # check filelist isn't empty
            if self.single_pass:
                filelist = sorted(filelist)
            else:
                random.shuffle(filelist)
            for f in filelist:
                num_files += 1
                reader = open(f, 'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes:
                        break  # finished reading this file
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack(
                        '%ds' % str_len, reader.read(str_len))[0]
                    yield tf.train.Example.FromString(example_str)
                if num_files % 1000 == 0:
                    print(('example_generator read {:d} files'.format(
                        num_files)), end='\r', flush=True)
            if self.single_pass:
                print("example_generator completed reading all datafiles. No more data.")
                break
        epoch += 1
        print(('Finished processing {:d} epoch(s)'.format(epoch)))

    def text_generator(self, example_gen):
        """Generates article and abstract text from tf.Example."""
        while True:
            e = six.next(example_gen)
            try:
                article_id = self._get_example_feature(e, 'article_id')
                article_text = self._get_example_feature(e, 'article_body')
                abstract_text = self._get_example_feature(
                    e, 'abstract')
                if not self._pubmed:
                  labels = self._get_example_feature(e, 'labels')
                section_names = self._get_example_feature(
                    e, 'section_names')
                sections = self._get_example_feature(e, 'sections')

                # convert to list
                article_text = _string_to_list(article_text)
                abstract_text = _string_to_list(abstract_text)
                if not self._pubmed:
                  labels = _string_to_list(labels, dtype='int')
                else:
                  labels = None
                section_names = _string_to_list(section_names)
                sections = _string_to_nested_list(sections)  # list of lists
            except ValueError:
                tf.logging.error(
                    'Failed to get article or abstract from example')
                continue
            if self.return_type == 'dict':
                yield {'article_id': article_id,
                       'article_text': article_text,
                       'abstract_text': abstract_text,
                       'labels': labels,
                       'section_names': section_names,
                       'sections': sections}
            else:
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
        
        
def read(path, pubmed=True):
  """ To import and use. returns a generator """
  bin_reader = BinReader(path,pubmed=pubmed)
  ex_gen = bin_reader.example_generator()
  input_gen = bin_reader.text_generator(ex_gen)
  counter = 0
  while True:
    try:
      ex = six.next(input_gen)
      if counter % 100 == 0:
        print('read {} records'.format(counter), end='\r')
      yield ex
    except StopIteration:
        print('done')
        break


def get_art_abs(article):
  abstract = LIST_SEPARATOR.join(article['abstract_text'])
  sections_str = _nested_list_to_string(article['sections'])
  section_names = LIST_SEPARATOR.join(article['section_names'])
  article_body = LIST_SEPARATOR.join(article['article_text'])
  try:
    labels = LIST_SEPARATOR.join([str(e) for e in article['labels']])
  except TypeError:
    labels = None
  return abstract, sections_str, section_names, article_body, labels


def write_to_bin(data, out_file, makevocab=False):

  pardir = str(Path(out_file).parent)
  if not os.path.exists(pardir):
    print('creating output directory {}'.format(pardir))
    os.mkdir(pardir)

  if makevocab:
    vocab_counter = collections.Counter()
  num_articles = len(data)
  with open(out_file, 'wb') as writer:
    for idx, s in enumerate(data):
      tf_example = tf.train.Example()
      abstract, sections_str, section_names, article_body, labels =\
          get_art_abs(s)
      tf_example.features.feature['article_id'].bytes_list.value.extend(
          [s['article_id'].encode('ascii', 'ignore')])
      tf_example.features.feature['abstract'].bytes_list.value.extend(
          [abstract.encode('ascii', 'ignore')])
      tf_example.features.feature['sections'].bytes_list.value.extend(
          [sections_str.encode('ascii', 'ignore')])
      tf_example.features.feature['section_names'].bytes_list.value.extend(
          [section_names.encode('ascii', 'ignore')])
      tf_example.features.feature['article_body'].bytes_list.value.extend(
          [article_body.encode('ascii', 'ignore')])
      try:
        tf_example.features.feature['labels'].bytes_list.value.extend(
            [labels.encode('ascii', 'ignore')])
      except AttributeError: # pubmed data doesn't have labels
        pass

      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      # Struct is used for packing data into strings
      # q is a long long interger (8 bytes)
      writer.write(struct.pack('q', str_len))
      # s format is bytes (char[])
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      if idx % 5 == 0:
        print('Finished writing {:.2f}\% of {:d} articles.'.format(
            idx * 100.0 / num_articles, num_articles), end='\r', flush=True)
      # Write the vocab to file, if applicable
#       if makevocab:
#         art_tokens = sections.split(' ')
#         art_tokens = [t for t in art_tokens if t not in [
#             SENTENCE_START, SENTENCE_END, SENTENCE_SEPARATOR, SECTION_SEPARATOR]]
#         abs_tokens = abstract.split(' ')
#         # remove these tags from vocab
#         abs_tokens = [t for t in abs_tokens if t not in [
#             SENTENCE_START, SENTENCE_END, SECTION_SEPARATOR, SENTENCE_SEPARATOR]]
#         tokens = art_tokens + abs_tokens
#         tokens = [t.strip() for t in tokens]  # strip
#         tokens = [t for t in tokens if t != ""]  # remove empty
#         vocab_counter.update(tokens)

  print('Finished writing {:d} articles.'.format(idx + 1))

  # write vocab to file
#   if makevocab:
#     log.info("Writing vocab file...")
#     with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
#       for word, count in vocab_counter.most_common(VOCAB_SIZE):
#         writer.write(word + ' ' + str(count) + '\n')
#     log.info("Finished writing vocab file")




def main(args):
    bin_reader = BinReader(data_path=args.data_path,pubmed=args.pubmed)
    ex_gen = bin_reader.example_generator()
    input_gen = bin_reader.text_generator(ex_gen)
    counter = 0
    added = 0
    res = []
    while True:
      try:
        ex = six.next(input_gen)
        counter += 1
        if args.filter:
          abst_len = len(' '.join(ex['abstract_text']).split())
          if abst_len < 50 or abst_len > 400:
            continue
          
          end_loc = len(ex['section_names'])
          beg_loc = 0
          if end_loc == 0:
            continue
          for i,s in enumerate(ex['section_names']):
            if 'conclu' in s.lower():
              end_loc = i + 1
            if 'intro' in s.lower() and beg_loc == 0:
              beg_loc = i
          sections = ex['sections']
          if beg_loc < len(ex['section_names']) - end_loc:
            sections = sections[beg_loc:end_loc]
          try:
            intro_last = sections[beg_loc][-2:] # last two sentences in the intro
          except IndexError:
  #             print('article_id: {}, len(sections): {}, section_names: {}'.format(article_id, len(sections), section_names))
            continue   
                 
          res.append(ex)
          added += 1
          if counter % 100 == 0:
            print('{} of {} will be added'.format(added, counter), end='\r')
      except StopIteration:
          print('{} of {} will be added'.format(added, counter))
          print('done')
          break
    random.seed(100)
    random.shuffle(res)
    if args.limit_size > 0:
      write_to_bin(res[:args.limit_size], args.out_file)
    else:
      write_to_bin(res, args.out_file)
       
       
def merge(paths, outpath, limit=3000):
  """ merge multiple bin files into a single bin file
  Args:
    paths: list of paths for bin files
    outpath: path to the output
  """
  res = []
  for path in paths:
    bin_reader = BinReader(data_path=path,pubmed=True)
    ex_gen = bin_reader.example_generator()
    input_gen = bin_reader.text_generator(ex_gen)
    counter1 = 0
    print('adding files from: ', path)
    while True:
      try:
        ex = six.next(input_gen)
        counter1 += 1
        res.append(ex)
        if counter1 % 100 == 0:
          print('{} will be added'.format(counter1), end='\r')
      except StopIteration:
          print('{} will be added'.format(counter1))
          break
  print('done')
  random.seed(200)
  random.shuffle(res)
  write_to_bin(res[:limit], outpath)
    

def divide(path, num=1000):
  res = []
  bin_reader = BinReader(data_path=path,pubmed=True)
  ex_gen = bin_reader.example_generator()
  input_gen = bin_reader.text_generator(ex_gen)
  counter1 = 0
  while True:
    try:
      ex = six.next(input_gen)
      counter1 += 1
      res.append(ex)
      if counter1 % 100 == 0:
        print('{} will be added'.format(counter1), end='\r', flush=True)
    except StopIteration:
        print('{} will be added'.format(counter1))
        break
  cur_path = pathlib.Path(path)
  name1 = str(cur_path).replace('.bin', '.part1.bin')
  name2 = str(cur_path).replace('.bin', '.part2.bin')
  print('writing first part with {} records in {}'.format(len(res) - num, name1))
  write_to_bin(res[num:], name1)
  print('writing second part with {} records in {}'.format(num, name2))
  write_to_bin(res[:num], name2)
    
    
        

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--data_path')
    ap.add_argument('--filter', action='store_true', default=False)
    ap.add_argument('--limit_size', type=int, default=0)
    ap.add_argument('--out_file')
    ap.add_argument('--pubmed', action='store_true', default=False)
    args = ap.parse_args()
    main(args)
#   merge(['/mnt/ilcompn0d1/user/cohan0/data/pubmed-tf-records/train-f.part2.bin',
#          '/mnt/ilcompn0d1/user/cohan0/data/pubmed-tf-records/test-f.bin'],
#         '/mnt/ilcompn0d1/user/cohan0/data/pubmed-tf-records/test-f-2000.bin')
         
