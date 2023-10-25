# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import itertools
import os

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from transformers import AutoTokenizer, AutoConfig, AutoModel
from aligner.word_align import SentenceAligner_word
from tqdm import tqdm




class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer, file_path_src, file_path_tgt, offsets=None):
        assert os.path.isfile(file_path_src)
        assert os.path.isfile(file_path_tgt)
        print('Loading the dataset...')
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path_src = file_path_src
        self.file_path_tgt = file_path_tgt
        self.offsets = offsets

    def process_line(self, line_src, line_tgt):
        """
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None
        """
        if len(line_src) == 0 or len(line_tgt) == 0:
            return None

        sent_src, sent_tgt = line_src.strip().split(), line_tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for
                                                                                 word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                       max_length=512)['input_ids'], \
                           self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                       max_length=512)['input_ids']


        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)

    def __iter__(self):

        f_src = open(self.file_path_src, encoding="utf-8")
        f_tgt = open(self.file_path_tgt, encoding="utf-8")
        i = 0
        for line_src, line_tgt in zip(f_src, f_tgt):
            i = i+1
            if line_src and line_tgt:

                processed = self.process_line(line_src, line_tgt)
                if processed is None:
                    print(
                        f'Line "{line_src.strip()}" (offset in bytes: {f_src.tell()}) is not in the correct format. Skipping...')
                    empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                    empty_sent = ''
                    yield (empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
                else:
                    yield processed






def word_align(args, tokenizer, model, folder_path, src_path, tgt_path):

    device = torch.device('cuda:1')
    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt


    dataset = LineByLineTextDataset(tokenizer, file_path_src=src_path, file_path_tgt=tgt_path)
    dataloader = DataLoader(
        dataset, batch_size=args.per_gpu_train_batch_size, collate_fn=collate
    )

    tqdm_iterator = trange(0, desc="Extracting")
    model_sentence = SentenceAligner_word(args, model)


    word_aligns_list_all_layer_dic = {}
    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch

            ids_src, ids_tgt = ids_src.to(args.device), ids_tgt.to(args.device)
            word_aligns_list_all_layer_dic_one_batch = model_sentence.get_aligned_word(args, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, output_prob = False)

            for layer_id in word_aligns_list_all_layer_dic_one_batch:

                if layer_id not in word_aligns_list_all_layer_dic:
                    word_aligns_list_all_layer_dic[layer_id] = word_aligns_list_all_layer_dic_one_batch[layer_id]
                else:
                    word_aligns_list_all_layer_dic[layer_id] = word_aligns_list_all_layer_dic[layer_id] + word_aligns_list_all_layer_dic_one_batch[layer_id]



    for layer_id in word_aligns_list_all_layer_dic:
        with open(os.path.join(folder_path, f'{"XX2XX.align"}.{str(layer_id)}'),'w', encoding='utf-8') as writers:
            for word_aligns in word_aligns_list_all_layer_dic[layer_id]:
                output_str = []
                for word_align in word_aligns:
                    if word_align[0] != -1:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                writers.write(' '.join(output_str) + '\n')


# def main():
#     parser = argparse.ArgumentParser()
#
#     # Required parameters
#     parser.add_argument(
#         "--data_file_src", default="/nfsshare/home/wangweikang/en2ces_101/output/en2Ces.src", type=str,
#         help="The input data file (a text file)."
#     )
#     parser.add_argument(
#         "--data_file_tgt", default="/nfsshare/home/wangweikang/en2ces_101/output/en2Ces.tgt", type=str,
#         help="The input data file (a text file)."
#     )
#     parser.add_argument(
#         "--output_file",
#         default='/nfsshare/home/wangweikang/my_alignment/valid_output/full/800',
#         type=str,
#         help="The output file."
#     )
#     parser.add_argument("--align_layer", type=int, default=6, help="layer for alignment extraction")
#     parser.add_argument(
#         "--extraction", default='softmax', type=str, help='softmax or others'
#     )
#     parser.add_argument(
#         "--softmax_threshold", type=float, default=0.1
#     )
#     parser.add_argument(
#         "--output_prob_file", default=None, type=str, help='The output probability file.'
#     )
#     parser.add_argument(
#         "--output_word_file", default=None, type=str, help='The output word file.'
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         default="/nfsshare/home/wangweikang/my_alignment/ckpt_output/full_fune/checkpoint-800",
#         type=str,
#         help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
#     )
#     parser.add_argument(
#         "--adapter_path",
#         default="/nfsshare/home/wangweikang/my_alignment/ckpt_output/full_fune/checkpoint-800",
#         type=str,
#         help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
#     )
#     parser.add_argument("--batch_size", default=32, type=int)
#
#     parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
#     parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
#     parser.add_argument("--tokenizer_name_or_path", type=str, default="xlm-roberta-base")
#     args = parser.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#     args.device = device
#
#
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#     model = AutoModel.from_pretrained(args.model_name_or_path)
#     if args.adapter_path:
#         model.load_adapter(args.adapter_path)
#         model.set_active_adapters('alignment_adapter')
#
#
#     word_align(args, tokenizer, model, args.output_file, args.data_file_src, args.data_file_tgt)
#
#
# if __name__ == "__main__":
#     main()

                    