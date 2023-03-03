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

- gold alignment
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

@inproceedings{wang-etal-2022-multilingual,
    title = "Multilingual Sentence Transformer as A Multilingual Word Aligner",
    author = "Wang, Weikang  and
      Chen, Guanhua  and
      Wang, Hanqing  and
      Han, Yue  and
      Chen, Yun",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.215",
    pages = "2952--2963",
    abstract = "Multilingual pretrained language models (mPLMs) have shown their effectiveness in multilingual word alignment induction. However, these methods usually start from mBERT or XLM-R. In this paper, we investigate whether multilingual sentence Transformer LaBSE is a strong multilingual word aligner. This idea is non-trivial as LaBSE is trained to learn language-agnostic sentence-level embeddings, while the alignment extraction task requires the more fine-grained word-level embeddings to be language-agnostic. We demonstrate that the vanilla LaBSE outperforms other mPLMs currently used in the alignment task, and then propose to finetune LaBSE on parallel corpus for further improvement. Experiment results on seven language pairs show that our best aligner outperforms previous state-of-the-art models of all varieties. In addition, our aligner supports different language pairs in a single model, and even achieves new state-of-the-art on zero-shot language pairs that does not appear in the finetuning process.",
}
