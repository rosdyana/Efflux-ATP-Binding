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

