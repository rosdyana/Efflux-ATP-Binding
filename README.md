# Efflux ATP Binding

## Usage 

## Preprocessing

### Generate PSSM using Blast
1. Make sure you've already the fasta files from Uniprot
2. Make sure you've already set blast tool and NR DB
3. run this command to generate pssm from fasta using blast
```
python pssmbyblast.py <fastadir>
```

### Generate position list of binding site
```
python generateposlist.py
```

### Filter pssm based on the similarity result
1. Make sure you've already run blastclust tool to generate similarity result
2. run this script to filter the pssm files
```
python filterpssm.py <resultsimilarityfile>
```

### Generate Libsvm from pssm files
1. run this script to generate libsvm files from pssm
```
python pssm_to_libsvm.py <folderofpssm> <windowssize>
```

### Generate csv from libsvm
1. run this script to generate csv as our final dataset 
```
python libsvm_to_csv.py <folderoflibsvm>
```

### Generate Dataset
1. run this script to generate dataset as training, validation and independent with customize ratio
```
python generatedataset.py <folderofcsv> <ratio>
```

### Run some algorithm
We provide some algorithm to try such as CNN, ResNet, VGG and DenseNet.

See this current result.

|Algo  |ACC  |TN  |FN  |TP  |FP  |Specitivity  |Sensitivity  |MCC  |AUC  |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|CNN  |100 %  |4177  |1  |48  |1  |100 %  |98 %  |0.979  |  |
|ResNet 18  |99 %  |4176  |1  |48  |2  |100 %  |98 %  |0.969  |  |
|ResNet 50  |99 %  |4176  |1  |48  |2 |100 %  |98 %  |0.969 |  |
|VGG 16  |  |  |  |  |  |  |  |  |  |
|VGG 19  |  |  |  |  |  |  |  |  |  |
|DenseNet  |  |  |  |  |  |  |  |  |  |