import spacy
import os
import glob
import gzip
import pickle
import random
import collections
import tensorflow as tf
import struct
import six
import numbers
import re
from itertools import chain

random.seed(200)

CHUNK_SIZE = 2000
VOCAB_SIZE = 50000


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


class BinWriter(object):
  
  def __init__(self, in_dir, out_dir, chunks_dir=None):
    self.chunks_dir = chunks_dir
    self.out_dir = out_dir
    self.in_dir = in_dir

  def _chunk_file(self, set_name):
    in_file = '%s/%s.bin' % (self.out_dir, set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
      chunk_fname = os.path.join(self.chunks_dir, '%s_%03d.bin' %
                                 (set_name, chunk))  # new chunk
      with open(chunk_fname, 'wb') as writer:
        for _ in range(CHUNK_SIZE):
          len_bytes = reader.read(8)
          if not len_bytes:
            finished = True
            break
          str_len = struct.unpack('q', len_bytes)[0]
          example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
          writer.write(struct.pack('q', str_len))
          writer.write(struct.pack('%ds' % str_len, example_str))
        chunk += 1


  def chunk_all(self):
    # Make a dir to hold the chunks
    if not os.path.isdir(self.chunks_dir):
      os.mkdir(self.chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
      print("Splitting %s data into chunks..." % set_name)
      self._chunk_file(set_name)
    print("Saved chunked data in %s" % self.chunks_dir)


  def _list_to_string(self, lst):
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


  def _nested_list_to_string(self, lst):
    ret = ''
    if not lst:
      return ret
    ret = SECTION_SEPARATOR.join(
        [LIST_SEPARATOR.join(e) for e in lst])
    return ret



  def write_to_bin(self, makevocab=True):
    indir, outdir = self.in_dir, self.out_dir
    nlp = spacy.load('en', vectors=False, entity=False)
    paths = glob.glob(indir + '/*.pkl.gz')
    random.shuffle(paths)
    total_files = len(paths)
    val_size = int(total_files * 0.04)
    test_size = 2000
    train_size = total_files - val_size - test_size
    train = paths[:train_size]
    val = paths[train_size:-test_size]
    test = paths[-test_size:]
    print('train size: {}, test size: {}, validation size: {}'.format(len(train), len(val), len(test)))
    sets = {'train': train, 'test': test, 'val': val}
      
    pardir = outdir
    if not os.path.exists(pardir):
      print('creating output directory {}'.format(pardir))
      os.mkdir(pardir)
  
    if makevocab:
      vocab_counter = collections.Counter()
  
    # iterate over training, test, and val sets
    for set_name in sets:
      print('processing {} set'.format(set_name))
      out_file = os.path.join(outdir, set_name) + '.bin'
      with open(out_file, 'wb') as writer:
        num_articles = len(sets[set_name])
        for idx, f in enumerate(sets[set_name]):
          with gzip.open(f) as f_:
            article_json = pickle.load(f_)
          tf_example = tf.train.Example()
  
          article_id = f.split('/')[-1].replace('.pkl.gz', '').encode('ascii', 'ignore')
          tf_example.features.feature['article_id'].bytes_list.value.extend([article_id])
          
#           abstract = article_json['abstract'].lower()
#           doc = nlp(abstract)
#           abstract_lst = [' '.join([e.text for e in sent]) for sent in doc.sents if sent]
#           abstract_lst = [' '.join([SENTENCE_START, sent, SENTENCE_END]) for sent in abstract_lst]
          
          # add abstract
          abstract = article_json['abstract']
          abstract_as_str = self._list_to_string(abstract)
          tf_example.features.feature['abstract'].bytes_list.value.extend([abstract_as_str.encode('utf-8', 'ignore') ])
          
#           sections = article_json['sections']
#           sections = [re.sub(r'\[\d+?(,\d+)*?\]', '', sect) for sect in sections]
#           new_sections = []
#           for sect in sections:
#             doc = nlp(sect)
#             new_sections.append([' '.join([e.text for e in sent]) for sent in doc.sents if sent and len(sent) > 15])
            
            
          # add sections, flatten sections (list of lists) as a string
          new_sections = article_json['sections']
          sections_str = self._nested_list_to_string(new_sections)
          tf_example.features.feature['sections'].bytes_list.value.extend([sections_str.encode('utf-8', 'ignore')])
          
          # add article body
          article_body = list(chain.from_iterable(new_sections))
          article_body_str = self._list_to_string(article_body)
          tf_example.features.feature['article_body'].bytes_list.value.extend([article_body_str.encode('utf-8', 'ignore')])
          
          # add section names
          section_names = [e if e else 'None' for e in article_json['section_names']]
          section_names = self._list_to_string(section_names)
          tf_example.features.feature['section_names'].bytes_list.value.extend([section_names.encode('utf-8', 'ignore')])  
          
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
          if makevocab:
            art_tokens = article_body_str.split(' ')
            art_tokens = [t for t in art_tokens
                          if t not in [
                SENTENCE_START, SENTENCE_END, SENTENCE_SEPARATOR,
                SECTION_SEPARATOR.strip(),
                LIST_SEPARATOR.strip()]]
            abs_tokens = article_body_str.split(' ')
            # remove these tags from vocab
            abs_tokens = [t for t in abs_tokens if t not in [
                SENTENCE_START, SENTENCE_END, SENTENCE_SEPARATOR,
                SECTION_SEPARATOR.strip(),
                LIST_SEPARATOR.strip()]]
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            vocab_counter.update(tokens)
  
        print("Finished writing file %s\n" % out_file)
  
    # write vocab to file
    if makevocab:
      print("Writing vocab file...")
      with open(os.path.join(self.out_dir, "vocab"), 'w') as writer:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
          writer.write(word + ' ' + str(count) + '\n')
      print("Finished writing vocab file")



if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--indir', '-i', help='path to the json_prepped_directory (.pkl.gz)',
                    required=True)
    ap.add_argument('--outdir', '-o', help='output directory to write bin files',
                    required=True)
    ap.add_argument('--chunksdir', '-c', help='output directory to write chunked bin files',
                    required=False)
    args = ap.parse_args()

    
    bin_writer = BinWriter(in_dir=args.indir, out_dir=args.outdir, chunks_dir=args.chunksdir )
    bin_writer.write_to_bin(makevocab=True)
#     bin_writer.chunk_all()