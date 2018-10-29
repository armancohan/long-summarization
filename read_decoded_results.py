import os
import sys

def main(path):
  fids = []
  for fname in os.listdir(path + '/decoded'):
    fid = fname.replace('_decoded.txt', '')
    fids.append(fid)
  for fid in fids:
    try:
      candidate = open(path + '/decoded/{}_decoded.txt'.format(fid)).read()
      ref = open(path + '/reference/{}_reference.txt'.format(fid)).read()
      print('Generated summary:\n{}\n\nReference summary:\n{}\n---\n.'.format(candidate, ref))
    except UnicodeDecodeError:
      print('WARN: ascii cannot decode.')

if __name__ == '__main__':
  main(sys.argv[1])
