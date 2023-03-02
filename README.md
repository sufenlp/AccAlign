# Multilingual Sentence Transformer as A Multilingual Word Aligner





## Requirements

Trained with Python 3.7, adapter-transformers 4.16.2, Torch 1.9.0, tqdm 4.62.3. 


## Data Format

- source
```
Wir glauben nicht , daß wir nur Rosinen herauspicken sollten .
Das stimmt nicht !
```

- target
```
We do not believe that we should cherry-pick .
But this is not what happens .
```

- golden alignments
```
9-8 8-8 7-8 6-6 1-1 2-2 4-5 5-5 3-3 11-9 10-7 2-4 
3-4 2-3 2-6 4-7 2-5 1-2 
```
## Directly extract alignments

```shell
bash run_align.sh
```


## Fine-tuning on training data

```shell
bash train.sh
```

## Calculate AER

```shell
bash cal_aer.sh
```

## Data 

Links to the test set used in the paper are here: 


| Language Pair  |   Type |Link |
| ------------- | ------------- | ------------- |
| En-De |   Gold Alignment | www-i6.informatik.rwth-aachen.de/goldAlignment/ |
| En-Fr |   Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt/ |
| En-Ro |   Gold Alignment | http://web.eecs.umich.edu/~mihalcea/wpt05/ |
| En-Fa |   Gold Alignment | https://ece.ut.ac.ir/en/web/nlp/resources |
| En-Zh |   Gold Alignment | https://nlp.csai.tsinghua.edu.cn/~ly/systems/TsinghuaAligner/TsinghuaAligner.html |
| En-Ja |   Gold Alignment | http://www.phontron.com/kftt |
| En-Sv |   Gold Alignment | https://www.ida.liu.se/divisions/hcs/nlplab/resources/ges/ |

Links to the training set and validation set used in the paper are here [here](https://drive.google.com/file/d/19X0mhTx6-EhgILm7_mtVWrT2qal-o-uV/view?usp=share_link)

## LaBSE

You can access to LaBSE model [here](https://huggingface.co/sentence-transformers/LaBSE) . 

## Adapter Checkpoints 

The multilingual word alignment adapter checkpoint [[download](https://drive.google.com/file/d/1eB8aWd4iM6DSQWJZOA5so4rB4MCQQyQf/view?usp=sharing)]


## Citation
