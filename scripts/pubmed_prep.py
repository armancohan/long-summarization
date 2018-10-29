import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
import sys
import glob
import json
from collections import defaultdict
import random
import pickle
import gzip
import os
import tqdm
from joblib import Parallel, delayed
import multiprocessing
import spacy
random.seed(10)

SENTENCE_START = '<S>'
SENTENCE_END = '</S>'

nlp = spacy.load('en', vectors=False, entity=False)

def get_sections(path):
    tree = ET.parse(path)
    root = tree.getroot()
    secs = root.findall('.//sec')
    section_names = []
    sections = []
    
    specific_type = False
    for s in secs:
      if 'sec-type' in s.attrib:
        specific_type = True
        break
    if specific_type:
      print([s.attrib for s in secs])
    try:
      abstract = BeautifulSoup(ET.tostring(root.find('.//abstract')),'xml').text
    except AttributeError:
      abstract = None
    try:
      title = root.find('.//article-title').text
    except AttributeError:
      title = None
    art_id = path.split('/')[-1][:-5]
    if len(secs) > 0 and 'id' in secs[0].attrib and re.search(r'^[Ss]ec\d\d?$', secs[0].attrib['id']):
      subsection = True
    else:
      subsection = False
    for sec in secs:
      if subsection:
        if 'id' not in sec.attrib:
          continue
        elif re.search(r'^[Ss]ec\d\d?$', sec.attrib['id']) is None:
          continue
      pars = sec.findall('.//p')
      this_section = []
      for p in pars:
              txt = str(ET.tostring(p), 'utf-8')
              txt = re.sub(r'<sup>.+?</sup>', '', txt)
              txt = BeautifulSoup(txt, 'xml').text
              if len(txt) > 20:
                  this_section.append(txt)
      try:
        section_names.append(sec.find('./title').text)
        sections.append(' '.join(this_section))
      except AttributeError:
        pass
    if len(secs) > 0 and len(sections) == 0:
      sections = []
    return art_id, title, abstract, sections, section_names

def process(path, outdir):
  fail = defaultdict(int)
  okay = True
  fpath = '{}/{}.pkl.gz'.format(outdir, path.split('/')[-1][:-5])
#   if os.path.exists(fpath):
#     return True
  res = get_sections(path)
  art_id, title, abstract, sections, section_names = res[0], res[1], res[2], res[3], res[4]
  if (path.split('/')[-1][:-5] != art_id):
    print(path.split('/')[-1][:-5], art_id)
    sys.exit()
  if len(sections) == 0:
    fail['sections'] += 1
    okay = False
  if abstract is None:
    fail['abstract'] += 1
    okay = False
  if title is None:
    fail['title'] += 1
    okay = False
  if section_names == [] or len(section_names) != len(sections):
    fail['sections'] += 1
    okay = False
  if okay:
    # write for json
          
    sections = [re.sub(r'\[\d+?(,\d+)*?\]', '', sect) for sect in sections]
    new_sections = []
    for sect in sections:
      doc = nlp(sect.lower())
      new_sections.append([' '.join([e.text for e in sent]) for sent in doc.sents if sent and len(sent) > 15])
    
    doc = nlp(abstract.lower())
    abstract_lst = [' '.join([e.text for e in sent]) for sent in doc.sents if sent]
    abstract_lst = [' '.join([SENTENCE_START, sent, SENTENCE_END]) for sent in abstract_lst]
      
    obj = {'article_id': art_id,
           'title': title,
           'abstract': abstract_lst,
           'sections': new_sections,
           'section_names': section_names}
    with gzip.open(fpath, 'w') as mf:
      pickle.dump(obj, mf)
    return True
  else:
    return False

def main(indir, outdir, logdir='/tmp/pubmed/'):
    cnt = 0
    fail = defaultdict(int)
    paths = glob.glob(indir + '/**/PMC4376923.nxml')
    total_files = len(paths)
    print('total xml files:' , total_files)
    if not os.path.exists(outdir):
      print('creating directory: ', outdir)
      os.makedirs(outdir)
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    num_cores = multiprocessing.cpu_count()
    res = Parallel(n_jobs=num_cores)(delayed(process)(fp, outdir)
                                 for fp in tqdm.tqdm(paths))
    num_success = len([e for e in res if e])
    num_err = len([e for e in res if not e])
    err = [paths[i] for i in range(len(paths)) if not res[i]]
    success = [paths[i] for i in range(len(paths)) if res[i]]
    with open(logdir + '/' + '.error.log', 'w') as mf:
      for elem in err:
        mf.write(elem + '\n')
    print('Done. Error: {} of {}'.format(num_err, num_err + num_success))
#     for path in tqdm.tqdm(paths):
#         cnt += 1
#         okay = True
#         res = get_sections(path)
#         art_id, title, abstract, sections, section_names = res[0], res[1], res[2], res[3], res[4]
#         if len(sections) == 0:
#           fail['sections'] += 1
#           okay = False
#         if abstract is None:
#           fail['abstract'] += 1
#           okay = False
#         if title is None:
#           fail['title'] += 1
#           okay = False
#         if cnt % 10 == 0:
#           total = sum([v for v in fail.values()])
#           print('{:.2f} processed; success: {:d} of {:d}, {:d} sections, {:d} abstract, {:d} title'.format(
#             (cnt*100/total_files), (cnt-total), cnt, fail['sections'], fail['abstract'], fail['title']), end='\r')
#         if okay:
#           # write for json
#           obj = {'article_id': art_id,
#                  'title': title,
#                  'abstract': abstract,
#                  'sections': sections,
#                  'section_names': section_names}
#           with gzip.open('{}/{}.pkl.gz'.format(outdir, art_id), 'w') as mf:
#             pickle.dump(obj, mf)
#         print(abstract[:100] + '...')
#     print(fail, cnt)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
