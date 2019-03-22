This repository contains data and code for the NAACL 2018 paper ["A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"](https://arxiv.org/abs/1804.05685). Please note that the code is not actively maintained.

#### Data

Two datasets of long and structured documents (scientific papers) are provided. The datasets are obtained from ArXiv and PubMed OpenAccess repositories.

ArXiv dataset: [Download](https://drive.google.com/file/d/1K2kDBTNXS2ikx9xKmi2Fy0Wsc5u_Lls0/view?usp=sharing)  
PubMed dataset: [Download](https://drive.google.com/file/d/1Sa3kip8IE0J1SkMivlgOwq1jBgOnzeny/view?usp=sharing)

The datasets are rather large. You need about 5G disk space to download and about 15G additional space when extracting the files. Each `tar` file consists of 4 files. `train.txt`, `val.txt`, `test.txt` respectively correspond to the training, validation, and test sets. These files are text files where each line is a json object corresponding to one scientific paper from ArXiv or PubMed. The `vocab` file is a plaintext file for the vocabulary. 

#### Code

The code is based on the pointer-generator network code by [See et al. (2017)](https://github.com/abisee/pointer-generator). Refer to their repo for documentation about the structure of the code.
You will need `python 3.6` and `Tensorflow 1.5` to run the code. The code might run with later versions of Tensorflow but it is not tested. Checkout other dependencies in `requirements.txt` file. A small sample of the dataset is already provided in this repo. To run the code with the sample data unzip the files in the `data` directory and simply execute the run script: `./run.sh`. To train the model with the entire dataset, first convert the jsonlines files to binary using the the following script: `scripts/json_to_bin.py` and modify the corresponding training data path in the `run.sh` script.

#### References

If you ended up finding this paper or repo useful please cite:
```
"A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"  
Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian  
NAACL-HLT 2018
```

Another relevant reference is Pointer-Generator network by See et al. (2017):
```
"Get to the point: Summarization with pointer-generator networks."  
Abigail See, Peter J. Liu, and Christopher D. Manning.  
ACL (2017).
``` 
