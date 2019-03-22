"""
Script to convert jsonlines files to tensorflow binary files
Usage: python json_to_bin.py   input.txt   output.bin
If you want to create vocabulary you can pass additional path to the output vocab file
python json_to_bin.py   input.txt    output.bin   --vocab_file   output.vocab
"""
import os
import glob
import gzip
import json
import random
import collections
import tensorflow as tf
import struct
import six
import numbers
import re
from itertools import chain
import pathlib

VOCAB_SIZE=50000

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote,
              dm_double_close_quote, ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<S>'
SENTENCE_END = '</S>'

# To represent list as string and retrieve it back
SENTENCE_SEPARATOR = ' <SENT/> '

# To represent list of sections as string and retrieve it back
SECTION_SEPARATOR = ' <SCTN/> '

# to represent separator as string, end of item (ei)
LIST_SEPARATOR = ' <EI/> '


def _list_to_string(lst):
  ret = ''
  if not lst:
    return ret
  if isinstance(lst[0], six.string_types):
    ret = LIST_SEPARATOR.join(lst)
  elif isinstance(lst[0], numbers.Number):
    ret = LIST_SEPARATOR.join([str(e) for e in lst])
  else:
    print(type(lst[0]))
    raise AttributeError('Unacceptable format of list to return to string')
  return ret


def _nested_list_to_string(lst):
  ret = ''
  if not lst:
    return ret
  ret = SECTION_SEPARATOR.join(
    [LIST_SEPARATOR.join(e) for e in lst])
  return ret


def write_to_bin(infile, outfile, vocab_file=False):
  pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
  writer = open(outfile, 'wb')
  if vocab_file:
    vocab_counter = collections.Counter()
  num_articles = sum([1 for _ in open(infile)])
  idx = 0
  for line in open(infile):
    idx += 1
    if not line.strip():
      continue
    line = line.strip()
    data = json.loads(line)
    tf_example = tf.train.Example()
    article_id = data['article_id'].encode('ascii', 'ignore')
    tf_example.features.feature['article_id'].bytes_list.value.extend([article_id])

    abstract_str = _list_to_string(data['abstract_text'])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract_str.encode('utf-8', 'ignore')])

    sections_str = _nested_list_to_string(data['sections'])
    tf_example.features.feature['sections'].bytes_list.value.extend([sections_str.encode('utf-8', 'ignore')])

    # add article body
    article_body = list(chain.from_iterable(data['sections']))
    article_body_str = _list_to_string(article_body)
    tf_example.features.feature['article_body'].bytes_list.value.extend([article_body_str.encode('utf-8', 'ignore')])

    # add section names
    section_names = [e if e else 'None' for e in data['section_names']]
    section_names = _list_to_string(section_names)
    tf_example.features.feature['section_names'].bytes_list.value.extend([section_names.encode('utf-8', 'ignore')])

    labels = _list_to_string(['0' for _ in range(len(data['section_names']))])
    tf_example.features.feature['labels'].bytes_list.value.extend([labels.encode('utf-8', 'ignore')])

    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    # Struct is used for packing data into strings
    # q is a long long interger (8 bytes)
    writer.write(struct.pack('q', str_len))
    # s format is bytes (char[])
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

    if idx % 5 == 0:
      print('Finished writing {:.3f}\% of {:d} articles.'.format(
        idx * 100.0 / num_articles, num_articles), end='\r', flush=True)

    # Write the vocab to file, if applicable
    if vocab_file:
      art_tokens = article_body_str.split(' ')
      art_tokens = [t for t in art_tokens
                    if t not in [
                      SENTENCE_START, SENTENCE_END, SENTENCE_SEPARATOR,
                      SECTION_SEPARATOR.strip(),
                      LIST_SEPARATOR.strip()]]
      abs_tokens = abstract_str.split(' ')
      # remove these tags from vocab
      abs_tokens = [t for t in abs_tokens if t not in [
        SENTENCE_START, SENTENCE_END, SENTENCE_SEPARATOR,
        SECTION_SEPARATOR.strip(),
        LIST_SEPARATOR.strip()]]
      tokens = art_tokens + abs_tokens
      tokens = [t.strip() for t in tokens]  # strip
      tokens = [t for t in tokens if t != ""]  # remove empty
      vocab_counter.update(tokens)


  print("Finished writing file %s\n" % outfile)
  writer.close()

  # write vocab to file
  if vocab_file:
    print("Writing vocab file...")
    with open(vocab_file, 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")



if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('infile', help='path to the jsonlines data')
    ap.add_argument('outfile', help='path to the output file')
    ap.add_argument('--vocab_file', help='path to the output vocabulary file'
                                         'this is optional, if not set it will not create'
                                         'vocab', default=False)
    args = ap.parse_args()

    write_to_bin(args.infile, args.outfile, args.vocab_file)
