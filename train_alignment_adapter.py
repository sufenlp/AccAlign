# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020, Zi-Yi Dou
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
import glob
import logging
import os
import random
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from self_training_modeling_adapter import BertForSO
from transformers import AutoTokenizer, AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup, HoulsbyConfig
from train_utils import _sorted_checkpoints, _rotate_checkpoints, WEIGHTS_NAME
from transformers import AdapterConfig


logger = logging.getLogger(__name__)

import itertools
from aligner.sent_aligner import word_align


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path_src, file_path_tgt, gold_path):
        assert os.path.isfile(file_path_src)
        assert os.path.isfile(file_path_tgt)
        logger.info("Creating features from dataset file at %s", file_path_src)

        assert file_path_src != file_path_tgt

        #cache_fn = f'{file_path_src}.cache' if gold_path is None else f'{file_path}.gold.cache'
        if args.cache_data and os.path.isfile(cache_fn) and not args.overwrite_cache:
            logger.info("Loading cached data from %s", cache_fn)
            self.examples = torch.load(cache_fn)
        else:
            # Loading text data
            self.examples = []
            with open(file_path_src, encoding="utf-8") as fs:
                lines_src = fs.readlines()
            with open(file_path_tgt, encoding="utf-8") as ft:
                lines_tgt = ft.readlines()

            # Loading gold data
            if gold_path is not None:
                assert os.path.isfile(gold_path)
                logger.info("Loading gold alignments at %s", gold_path)
                with open(gold_path, encoding="utf-8") as f:
                    gold_lines = f.readlines()
                assert len(gold_lines) == len(lines_src)

            i = 0
            for line_id, (line_src, line_tgt) in tqdm(enumerate(zip(lines_src, lines_tgt))):
                i = i + 1
                if line_src and line_tgt:
                    sent_src, sent_tgt = line_src.strip().split(), line_tgt.strip().split()
                    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word)
                                                                                             for word in sent_tgt]
                    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [
                        tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
                    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                                   max_length=args.max_len)['input_ids'], \
                                       tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                                   max_length=args.max_len)['input_ids']


                    if len(ids_src) == 2 or len(ids_tgt) == 2:
                        #logger.info("Skipping instance src %s", line_src)
                        #logger.info("Skipping instance tgt %s", lines_tgt)
                        continue
                        


                    bpe2word_map_src = []
                    for i, word_list in enumerate(token_src):
                        bpe2word_map_src += [i for x in word_list]
                    bpe2word_map_tgt = []
                    for i, word_list in enumerate(token_tgt):
                        bpe2word_map_tgt += [i for x in word_list]

                    if gold_path is not None:
                        try:
                            gold_line = gold_lines[line_id].strip().split()
                            gold_word_pairs = []
                            for src_tgt in gold_line:
                                if 'p' in src_tgt:
                                    if args.ignore_possible_alignments:
                                        continue
                                    wsrc, wtgt = src_tgt.split('p')
                                else:
                                    wsrc, wtgt = src_tgt.split('-')
                                wsrc, wtgt = (int(wsrc), int(wtgt)) if not args.gold_one_index else (
                                    int(wsrc) - 1, int(wtgt) - 1)
                                gold_word_pairs.append((wsrc, wtgt))
                            self.examples.append(
                                (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, gold_word_pairs,[len(ids_src)-2, len(ids_tgt)-2]))
                        except:
                            logger.info("Error when processing the gold alignment %s, skipping",
                                        gold_lines[line_id].strip())
                            continue
                    else:
                        self.examples.append((ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, None, [len(ids_src)-2, len(ids_tgt)-2]))

            if args.cache_data:
                logger.info("Saving cached data to %s", cache_fn)
                torch.save(self.examples, cache_fn)



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        neg_i = random.randint(0, len(self.examples) - 1)
        while neg_i == i:
            neg_i = random.randint(0, len(self.examples) - 1)
        return tuple(list(self.examples[i]) + list(self.examples[neg_i][:2]))


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path_src = args.eval_data_file_src if evaluate else args.train_data_file_src
    file_path_tgt = args.eval_data_file_tgt if evaluate else args.train_data_file_tgt
    gold_path = args.eval_gold_file if evaluate else args.train_gold_file
    return LineByLineTextDataset(tokenizer, args, file_path_src=file_path_src, file_path_tgt=file_path_tgt, gold_path=gold_path)


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)





def train(args, train_dataset, model, tokenizer) -> Tuple[int, float]:
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)



    def collate(examples):
        #model_init.eval()
        model.eval()
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        word_aligns = []
        pairs_len = []
        for example in examples:
            end_id = example[0][-1].view(-1)

            src_id = example[0][:args.block_size]
            src_id = torch.cat([src_id[:-1], end_id])
            tgt_id = example[1][:args.block_size]
            tgt_id = torch.cat([tgt_id[:-1], end_id])

            examples_src.append(src_id)
            examples_tgt.append(tgt_id)
            src_len = max(src_len, len(src_id))
            tgt_len = max(tgt_len, len(tgt_id))

            bpe2word_map_src.append(example[2])
            bpe2word_map_tgt.append(example[3])
            word_aligns.append(example[4])

            pairs_len.append(example[5])

        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)


        if word_aligns[0] is None:
            word_aligns = None
        if args.n_gpu > 1 or args.local_rank != -1:

            guides = model.get_aligned_word(examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt,
                                                   args.device, src_len, tgt_len, align_layer=args.align_layer,
                                                   extraction=args.extraction, softmax_threshold=args.softmax_threshold,
                                                   word_aligns=word_aligns, pairs_len=pairs_len)
        else:
            guides = model.get_aligned_word(examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device,
                                            src_len, tgt_len, align_layer=args.align_layer, extraction=args.extraction,
                                            softmax_threshold=args.softmax_threshold, word_aligns=word_aligns, pairs_len=pairs_len)

        return examples_src, examples_tgt, guides

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )


    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.max_steps > 0 and args.max_steps < t_total:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (not (any(nd in n for nd in no_decay)))],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if ((any(nd in n for nd in no_decay)))],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    # Check if continuing training from a checkpoint
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility

    #writer_train = open(args.train_res_dir, "w")


    def backward_loss(loss, tot_loss):
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        tot_loss += loss.item()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        return tot_loss

    tqdm_iterator = trange(int(t_total), desc="Iteration", disable=args.local_rank not in [-1, 0])
    for _ in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()

            if args.train_so:
                inputs_src, inputs_tgt = batch[0].clone(), batch[1].clone()
                inputs_src, inputs_tgt = inputs_src.to(args.device), inputs_tgt.to(args.device)
                attention_mask_src, attention_mask_tgt = (inputs_src != 0), (inputs_tgt != 0)
                guide = batch[2].to(args.device)
                loss = model(inputs_src=inputs_src, inputs_tgt=inputs_tgt, attention_mask_src=attention_mask_src,
                             attention_mask_tgt=attention_mask_tgt, guide=guide, align_layer=args.align_layer,
                             extraction=args.extraction, softmax_threshold=args.softmax_threshold,
                             )
                tr_loss = backward_loss(loss, tr_loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                tqdm_iterator.update()

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("  Step %s. Training loss = %s", str(global_step),
                                str((tr_loss - logging_loss) / args.logging_steps))


                    logger.info("***** Training results {} *****".format(global_step))
                    #for key in sorted(result.keys()):
                    logger.info("  %s = %s", str(global_step)+' steps', str((tr_loss - logging_loss) / args.logging_steps))
                    #writer_train.write("%s = %s\n" % (str(global_step)+' steps', str((tr_loss - logging_loss) / args.logging_steps)))

                    logging_loss = tr_loss

                    evaluate(args, model, tokenizer, global_step, prefix='')


                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"



                    output_dir_adapter = os.path.join(args.output_dir_adapter, "{}-{}".format(checkpoint_prefix, global_step))

                    model.save_adapter(output_dir_adapter, "alignment_adapter")
                    logger.info("Saving adapters to %s", output_dir_adapter)

            if global_step > t_total:
                break

        if global_step > t_total:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, global_step, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.eval_res_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    def collate(examples):
        model.eval()
        examples_src, examples_tgt, examples_srctgt, examples_tgtsrc, langid_srctgt, langid_tgtsrc, psi_examples_srctgt, psi_labels = [], [], [], [], [], [], [], []
        src_len = tgt_len = 0
        bpe2word_map_src, bpe2word_map_tgt = [], []
        word_aligns = []
        pairs_len = []
        for example in examples:
            end_id = example[0][-1].view(-1)

            src_id = example[0][:args.block_size]
            src_id = torch.cat([src_id[:-1], end_id])
            tgt_id = example[1][:args.block_size]
            tgt_id = torch.cat([tgt_id[:-1], end_id])

            examples_src.append(src_id)
            examples_tgt.append(tgt_id)
            src_len = max(src_len, len(src_id))
            tgt_len = max(tgt_len, len(tgt_id))

            bpe2word_map_src.append(example[2])
            bpe2word_map_tgt.append(example[3])
            word_aligns.append(example[4])
            pairs_len.append(example[5])

        examples_src = pad_sequence(examples_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        examples_tgt = pad_sequence(examples_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)

        if word_aligns[0] is None:
            word_aligns = None

        guides = model.get_aligned_word(examples_src, examples_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device,
                                        src_len, tgt_len, align_layer=args.align_layer, extraction=args.extraction,
                                        softmax_threshold=args.softmax_threshold, test=False, word_aligns=word_aligns,pairs_len=pairs_len)

        return examples_src, examples_tgt, guides, bpe2word_map_src, bpe2word_map_tgt

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    set_seed(args)  # Added here for seeds



    folder_path = os.path.join(args.eval_res_dir, str(global_step) + '_steps')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    output_file = os.path.join(folder_path, "dev.align.6")

    writers = open(output_file, 'w', encoding='utf-8')


    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs_src, inputs_tgt = batch[0].clone(), batch[1].clone()
            inputs_src, inputs_tgt = inputs_src.to(args.device), inputs_tgt.to(args.device)
            attention_mask_src, attention_mask_tgt = (inputs_src != 0), (inputs_tgt != 0)

            bpe2word_map_src, bpe2word_map_tgt = batch[3], batch[4]


            word_aligns_list_batch = model.get_aligned_word(inputs_src, inputs_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, [],
                             [],
                             align_layer=6, extraction='softmax', softmax_threshold=0.1, test=True,
                             output_prob=False,
                             word_aligns=None, pairs_len=None)


            for aligns_set in word_aligns_list_batch:
                output_str = []
                for word_align in aligns_set:
                    if word_align[0] != -1:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                writers.write(' '.join(output_str) + '\n')





def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--eval_data_file_src", default='', type=str,  help="The input evluating data file (a text file)."
    )
    parser.add_argument(
        "--eval_data_file_tgt", default="", type=str,  help="The input evaluating data file (a text file)."
    )
    
    parser.add_argument(
        "--train_data_file_src", default='', type=str,
         help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--train_data_file_tgt", default="", type=str,
         help="The input training data file (a text file)."
    )

    parser.add_argument(
        "--infer_data_file_src", default='', type=str, help="The input evluating data file (a text file)."
    )
    parser.add_argument(
        "--infer_data_file_tgt", default="", type=str, help="The input evaluating data file (a text file)."
    )


    
    
    parser.add_argument(
        "--output_dir",
        default='',
        type=str,
        #required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_dir_adapter",
        default='',
        type=str,
        # required=True,
        help="The output directory where the adapters will be written.",
    )
    parser.add_argument(
        "--train_res_dir",
        default='',
        type=str,
        # required=True,
        help="The output directory where the training loss will be written.",
    )
    parser.add_argument(
        "--eval_res_dir",
        default='',
        type=str,
        # required=True,
        help="The output directory where the eval loss will be written.",
    )
    parser.add_argument(
        "--infer_path",
        default='',
        type=str,
        # required=True,
        help="The output directory where the inference results will be written.",
    )



    parser.add_argument("--train_so", action="store_true")
    # Supervised settings
    parser.add_argument(
        "--train_gold_file", default=None, type=str, help="Gold alignment for training data"
    )
    parser.add_argument(
        "--eval_gold_file", default=None, type=str, help="Gold alignment for evaluation data"
    )
    parser.add_argument(
        "--ignore_possible_alignments", action="store_true", help="Whether to ignore possible gold alignments"
    )
    parser.add_argument(
        "--gold_one_index", action="store_true", help="Whether the gold alignment files are one-indexed"
    )
    # Other parameters
    parser.add_argument("--cache_data", action="store_true", help='if cache the dataset')
    parser.add_argument("--align_layer", type=int, default=6, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, choices=['softmax', 'entmax'], help='softmax or entmax'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.1
    )

    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--adapter_path",
        default=None,
        type=str,
        help="The adapter checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    # parser.add_argument(
    #     "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    # )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--max_len",
        default=512,
        type=int,
        help="max sequence length"
                 )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run infer on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set the maximum number of training steps to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=25, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument("--reduction_factor", type=int, default=6, help="reduction_factor for adapter")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    if args.eval_data_file_src is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir_adapter)
            and os.listdir(args.output_dir_adapter)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir_adapter
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    #config_class, model_class, tokenizer_class = BertConfig, BertForSO, BertTokenizer
    modelforALING, tokenizer_class = BertForSO, AutoTokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = args.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, args.max_len)


    labse_model = AutoModel.from_pretrained(args.model_name_or_path, output_hidden_states=True)

    if args.do_train:

        config_align = HoulsbyConfig(reduction_factor=6)

        labse_model.add_adapter("alignment_adapter", config=config_align)
        labse_model.train_adapter("alignment_adapter")
        labse_model.set_active_adapters("alignment_adapter")


        model = modelforALING(args, config, labse_model)

        model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)



        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)






    if args.do_test:
        # extract word alignment for all layers
        if args.adapter_path:
            labse_model.load_adapter(args.adapter_path)
            labse_model.set_active_adapters('alignment_adapter')
        model = modelforALING(args, config, labse_model)
        model.to(args.device)

        # folder_path= os.path.join(args.infer_path, f'{}2{}' + '_steps')
        folder_path = args.infer_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        word_align(args, tokenizer, model, folder_path, args.infer_data_file_src,
                   args.infer_data_file_tgt)




if __name__ == "__main__":
    main()







