import transformers
from transformers import AutoModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from transformers import PreTrainedModel


PAD_ID=0
CLS_ID=101
SEP_ID=102

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


class ModelGuideHead(nn.Module):
    def __init__(self):
        super().__init__()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_src, hidden_states_tgt,
        inputs_src, inputs_tgt,
        guide=None,
        extraction='softmax', softmax_threshold=0.1,
        output_prob=False,
    ):
        #mask
        attention_mask_src = ( (inputs_src==PAD_ID) + (inputs_src==CLS_ID) + (inputs_src==SEP_ID) ).float()
        attention_mask_tgt = ( (inputs_tgt==PAD_ID) + (inputs_tgt==CLS_ID) + (inputs_tgt==SEP_ID) ).float()
        len_src = torch.sum(1-attention_mask_src, -1)
        len_tgt = torch.sum(1-attention_mask_tgt, -1)
        attention_mask_src = return_extended_attention_mask(1-attention_mask_src, hidden_states_src.dtype)
        attention_mask_tgt = return_extended_attention_mask(1-attention_mask_tgt, hidden_states_src.dtype)

        #qkv
        query_src = self.transpose_for_scores(hidden_states_src)
        query_tgt = self.transpose_for_scores(hidden_states_tgt)
        key_src = query_src
        key_tgt = query_tgt
        value_src = query_src
        value_tgt = query_tgt

        #att
        attention_scores = torch.matmul(query_src, key_tgt.transpose(-1, -2))
        attention_scores_src = attention_scores + attention_mask_tgt
        attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)


        attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src) #if extraction == 'softmax' else entmax15(attention_scores_src, dim=-1)
        attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt) #if extraction == 'softmax' else entmax15(attention_scores_tgt, dim=-2)



        if guide is None:

            threshold = softmax_threshold if extraction == 'softmax' else 0
            align_matrix = (attention_probs_src>threshold)*(attention_probs_tgt>threshold)
            if not output_prob:
                return align_matrix
            # A heuristic of generating the alignment probability
            attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src/torch.sqrt(len_tgt.view(-1, 1, 1, 1)))
            attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt/torch.sqrt(len_src.view(-1, 1, 1, 1)))
            align_prob = (2*attention_probs_src*attention_probs_tgt)/(attention_probs_src+attention_probs_tgt+1e-9)
            return align_matrix, align_prob




        so_loss_src = torch.sum(torch.sum (attention_probs_src*guide, -1), -1).view(-1)
        so_loss_tgt = torch.sum(torch.sum (attention_probs_tgt*guide, -1), -1).view(-1)

        so_loss = so_loss_src/len_src + so_loss_tgt/len_tgt
        so_loss = -torch.mean(so_loss)



        return so_loss





class BertForSO(PreTrainedModel):
    def __init__(self, args, config, model_adapter):
        super().__init__(config)
        self.model = model_adapter
        self.guide_layer = ModelGuideHead()

    def forward(
            self,
            inputs_src,
            inputs_tgt=None,
            labels_src=None,
            labels_tgt=None,
            attention_mask_src=None,
            attention_mask_tgt=None,
            align_layer=6,
            guide=None,
            extraction='softmax', softmax_threshold=0.1,
            position_ids1=None,
            position_ids2=None,
            do_infer=False,
    ):

        loss_fct =CrossEntropyLoss(reduction='none')
        batch_size = inputs_src.size(0)

        output_src = self.model(
            inputs_src,
            attention_mask=attention_mask_src,
            position_ids=position_ids1,
        )

        
        output_tgt = self.model(
            inputs_tgt,
            attention_mask=attention_mask_tgt,
            position_ids=position_ids2,
        )
        if do_infer:
            return output_src, output_tgt
        
        if guide is None:
            raise ValueError('must specify labels for the self-trianing objective')

        

        hidden_states_src = output_src.hidden_states[align_layer]
        hidden_states_tgt = output_tgt.hidden_states[align_layer]


        sco_loss = self.guide_layer(hidden_states_src, hidden_states_tgt, inputs_src, inputs_tgt, guide=guide,
                                    extraction=extraction, softmax_threshold=softmax_threshold)
        return sco_loss

    def save_adapter(self, save_directory, adapter_name):
        self.model.save_adapter(save_directory, adapter_name)
        
    def get_aligned_word(self, inputs_src, inputs_tgt, bpe2word_map_src, bpe2word_map_tgt, device, src_len, tgt_len,
                         align_layer=6, extraction='softmax', softmax_threshold=0.1, test=False, output_prob=False,
                         word_aligns=None, pairs_len=None):
        batch_size = inputs_src.size(0)
        bpelen_src, bpelen_tgt = inputs_src.size(1) - 2, inputs_tgt.size(1) - 2
        if word_aligns is None:
            inputs_src = inputs_src.to(dtype=torch.long, device=device).clone()
            inputs_tgt = inputs_tgt.to(dtype=torch.long, device=device).clone()

            with torch.no_grad():
                outputs_src = self.model(
                    inputs_src,
                    attention_mask=(inputs_src != PAD_ID),
                )
                outputs_tgt = self.model(
                    inputs_tgt,
                    attention_mask=(inputs_tgt != PAD_ID),
                )


                hidden_states_src = outputs_src.hidden_states[align_layer]
                hidden_states_tgt = outputs_tgt.hidden_states[align_layer]

                attention_probs_inter = self.guide_layer(hidden_states_src, hidden_states_tgt, inputs_src, inputs_tgt,
                                                         extraction=extraction, softmax_threshold=softmax_threshold,
                                                         output_prob=output_prob)
                if output_prob:
                    attention_probs_inter, alignment_probs = attention_probs_inter
                    alignment_probs = alignment_probs[:, 0, 1:-1, 1:-1]
                attention_probs_inter = attention_probs_inter.float()

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

        if test:
            
            return word_aligns



        guide = torch.zeros(batch_size, 1, src_len, tgt_len)
        for idx, (word_align, b2w_src, b2w_tgt) in enumerate(zip(word_aligns, bpe2word_map_src, bpe2word_map_tgt)):
            len_src = min(bpelen_src, len(b2w_src))
            len_tgt = min(bpelen_tgt, len(b2w_tgt))

            for i in range(len_src):
                for j in range(len_tgt):
                    if (b2w_src[i], b2w_tgt[j]) in word_align:
                        guide[idx, 0, i + 1, j + 1] = 1.0



        return guide
