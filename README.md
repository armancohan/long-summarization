
#### Data

Two datasets of long and structured documents (scientific papers) are provided. The datasets are obtained from ArXiv and PubMed OpenAccess repositories.

ArXiv dataset: [Download](https://drive.google.com/file/d/1ABdvyNdTv0eTqHNB0zk_n7tJsiN3gkMg/view?usp=sharing)  
PubMed dataset: [Download](https://drive.google.com/file/d/1zfS7Yaw76o0p1vL-LJC_7AdsUZtztmtR/view?usp=sharing)

The datasets are rather large. You need about 5G disk space to download and about 15G additional space when extracting the files. Each `tar` file consists of 4 files. `train.txt`, `val.txt`, `test.txt` respectively correspond to the training, validation, and test sets. These files are text files where each line is a json object corresponding to one scientific paper from ArXiv or PubMed. The `vocab` file is a plaintext file for the vocabulary. 

#### Reference

```
"A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"
Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian  
NAACL-HLT 2018
```
