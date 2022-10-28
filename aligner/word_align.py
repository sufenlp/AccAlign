# coding=utf-8

import os
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from transformers import BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, \
    XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
from train_utils import get_logger

LOG = get_logger(__name__)


def return_extended_attention_mask(attention_mask, dtype):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids or attention_mask"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask



class SentenceAligner_word(object):
    def __init__(self, args, model):

        self.guide = None
        self.softmax_threshold = args.softmax_threshold
        self.embed_loader = model

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_subword_matrix(self, args, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID, output_prob=False):

        output_src,output_tgt = self.embed_loader(
            inputs_src=inputs_src, inputs_tgt=inputs_tgt, attention_mask_src=(inputs_src != PAD_ID),
            attention_mask_tgt=(inputs_tgt != PAD_ID), guide=None, align_layer=args.align_layer,
            extraction=args.extraction, softmax_threshold=args.softmax_threshold,
            train_so=args.train_so, train_co=args.train_co, do_infer=True,
        )

        align_matrix_all_layers = {}

        for layer_id in range(1, len(output_src.hidden_states)):

            hidden_states_src = output_src.hidden_states[layer_id]
            hidden_states_tgt = output_tgt.hidden_states[layer_id]
            # mask
            attention_mask_src = ((inputs_src == PAD_ID) + (inputs_src == CLS_ID) + (inputs_src == SEP_ID)).float()
            attention_mask_tgt = ((inputs_tgt == PAD_ID) + (inputs_tgt == CLS_ID) + (inputs_tgt == SEP_ID)).float()
            len_src = torch.sum(1 - attention_mask_src, -1)
            len_tgt = torch.sum(1 - attention_mask_tgt, -1)
            attention_mask_src = return_extended_attention_mask(1 - attention_mask_src, hidden_states_src.dtype)
            attention_mask_tgt = return_extended_attention_mask(1 - attention_mask_tgt, hidden_states_tgt.dtype)

            # qkv
            query_src = self.transpose_for_scores(hidden_states_src)
            query_tgt = self.transpose_for_scores(hidden_states_tgt)
            key_src = query_src
            key_tgt = query_tgt
            value_src = query_src
            value_tgt = query_tgt

            # att
            attention_scores = torch.matmul(query_src, key_tgt.transpose(-1, -2))
            attention_scores_src = attention_scores + attention_mask_tgt
            attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)

            attention_probs_src = nn.Softmax(dim=-1)(
                attention_scores_src)  # if extraction == 'softmax' else entmax15(attention_scores_src, dim=-1)
            attention_probs_tgt = nn.Softmax(dim=-2)(
                attention_scores_tgt)  # if extraction == 'softmax' else entmax15(attention_scores_tgt, dim=-2)

            if self.guide is None:
                # threshold = softmax_threshold if extraction == 'softmax' else 0
                threshold = self.softmax_threshold
                align_matrix = (attention_probs_src > threshold) * (attention_probs_tgt > threshold)

                if not output_prob:
                    # return align_matrix
                    align_matrix_all_layers[layer_id] = align_matrix
                # A heuristic of generating the alignment probability
                """
                attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src/torch.sqrt(len_tgt.view(-1, 1, 1, 1)))
                attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt/torch.sqrt(len_src.view(-1, 1, 1, 1)))
                align_prob = (2*attention_probs_src*attention_probs_tgt)/(attention_probs_src+attention_probs_tgt+1e-9)
                return align_matrix, align_prob
                """

        return align_matrix_all_layers

    def get_aligned_word(self, args, inputs_src, inputs_tgt, bpe2word_map_src, bpe2word_map_tgt, PAD_ID, CLS_ID, SEP_ID,
                         output_prob=False):

        attention_probs_inter_all_layers = self.get_subword_matrix(args, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID,
                                                                   output_prob)
        if output_prob:
            attention_probs_inter, alignment_probs = attention_probs_inter
            alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]

        word_aligns_all_layers = {}

        for layer_id in attention_probs_inter_all_layers:

            attention_probs_inter = attention_probs_inter_all_layers[layer_id].float()

            word_aligns = []
            attention_probs_inter = attention_probs_inter[:, 0, 1:-1, 1:-1]

            for idx, (attention, b2w_src, b2w_tgt) in enumerate(
                    zip(attention_probs_inter, bpe2word_map_src, bpe2word_map_tgt)):
                aligns = set() if not output_prob else dict()
                non_zeros = torch.nonzero(attention)
                for i, j in non_zeros:
                    word_pair = (b2w_src[i], b2w_tgt[j])
                    if output_prob:
                        prob = alignment_probs[idx, i, j]
                        if not word_pair in aligns:
                            aligns[word_pair] = prob
                        else:
                            aligns[word_pair] = max(aligns[word_pair], prob)
                    else:
                        aligns.add(word_pair)
                word_aligns.append(aligns)

            word_aligns_all_layers[layer_id] = word_aligns
        return word_aligns_all_layers
