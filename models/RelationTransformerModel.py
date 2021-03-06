##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
#utils.PositionalEmbedding()
import copy
import math
import numpy as np

base = '/home/xilini/object_relation_transformer/data/representations/'
img_num = 0
seq_actual_length = 0

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator


    def forward(self, src, boxes, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, boxes, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, boxes, src_mask):
        return self.encoder(self.src_embed(src), boxes, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, box, mask):
        "Pass the input (and mask) through each layer in turn."
                
        multihead_enc_token_emb = torch.Tensor(()).to(device='cuda:0')
        multihead_enc_context_emb = torch.Tensor(()).to(device='cuda:0')
        multihead_enc_attn = torch.Tensor(()).to(device='cuda:0')        
        
        for layer in self.layers:
                        
            x,\
            enc_token_emb,\
            enc_context_emb,\
            enc_attn = layer(x, box, mask)
            
            if enc_token_emb is not None:
                multihead_enc_token_emb = torch.cat((multihead_enc_token_emb, enc_token_emb), 0)
                multihead_enc_context_emb = torch.cat((multihead_enc_context_emb, enc_context_emb), 0)
                multihead_enc_attn = torch.cat((multihead_enc_attn, enc_attn), 0)
                x = x

            else:
                x = x
                
        #print(multihead_enc_token_emb.shape)
        if multihead_enc_token_emb.shape[0] == 6: # collected all layers
            torch.save(multihead_enc_attn, base + f'train_enc_attn/{img_num}.pt')
            torch.save(multihead_enc_context_emb, base + f'train_enc_context_emb/{img_num}.pt')
            torch.save(multihead_enc_token_emb, base + f'train_enc_token_emb/{img_num}.pt')
            #print(multihead_enc_attn.shape)
            #print(multihead_enc_context_emb.shape)
            #print(multihead_enc_token_emb.shape)
                              
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, box, mask):
        #print('ENCODER LAYER')
        "Follow Figure 1 (left) for connections."
        
        enc_token_emb, enc_context_emb, enc_attn = self.self_attn(x, x, x, box, mask, one_out=None)
        
        if enc_context_emb is not None and enc_attn is not None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, box, mask, one_out=True))
            return self.sublayer[1](x, self.feed_forward),\
                   enc_token_emb,\
                   enc_context_emb,\
                   enc_attn
        
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, box, mask, one_out=True))
            return self.sublayer[1](x, self.feed_forward),\
                   None, None, None

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        
        global img_num
        
        multihead_dec_token_emb = torch.Tensor(()).to(device='cuda:0')
        multihead_dec_context_emb = torch.Tensor(()).to(device='cuda:0')
        multihead_dec_attn = torch.Tensor(()).to(device='cuda:0')
        multihead_encdec_token_emb = torch.Tensor(()).to(device='cuda:0')
        multihead_encdec_context_emb = torch.Tensor(()).to(device='cuda:0')
        multihead_encdec_attn = torch.Tensor(()).to(device='cuda:0')
        
        
        for layer in self.layers:
            
            x,\
            dec_token_emb,\
            dec_context_emb,\
            dec_attn,\
            encdec_token_emb,\
            encdec_context_emb,\
            encdec_attn = layer(x, memory, src_mask, tgt_mask)
            
            if dec_token_emb is not None:
                
                multihead_dec_token_emb = torch.cat((multihead_dec_token_emb, dec_token_emb.unsqueeze(0)), 0)
                multihead_dec_context_emb = torch.cat((multihead_dec_context_emb, dec_context_emb.unsqueeze(0)), 0)
                multihead_dec_attn = torch.cat((multihead_dec_attn, dec_attn.unsqueeze(0)), 0)
                multihead_encdec_token_emb = torch.cat((multihead_encdec_token_emb, encdec_token_emb.unsqueeze(0)), 0)
                multihead_encdec_context_emb = torch.cat((multihead_encdec_context_emb, encdec_context_emb.unsqueeze(0)), 0)
                multihead_encdec_attn = torch.cat((multihead_encdec_attn, encdec_attn.unsqueeze(0)), 0)
                x = x

            else:
                x = x
                
        #print(multihead_dec_token_emb.shape)
        if multihead_dec_token_emb.shape[0] == 6: # collected all layers
            torch.save(multihead_dec_attn, base + f'train_dec_attn/{img_num}.pt')
            torch.save(multihead_dec_context_emb, base + f'train_dec_context_emb/{img_num}.pt')
            torch.save(multihead_dec_token_emb, base + f'train_dec_token_emb/{img_num}.pt')
            torch.save(multihead_encdec_attn, base + f'train_encdec_attn/{img_num}.pt')
            torch.save(multihead_encdec_context_emb, base + f'train_encdec_context_emb/{img_num}.pt')
            torch.save(multihead_encdec_token_emb, base + f'train_encdec_token_emb/{img_num}.pt')
            img_num += 1
            #print(multihead_dec_attn.shape)
            #print(multihead_dec_context_emb.shape)
            #print(multihead_dec_token_emb.shape)
            #print(multihead_encdec_attn.shape)
            #print(multihead_encdec_context_emb.shape)
            #print(multihead_encdec_token_emb.shape)
            #print(img_num)
                
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        #print('DECODER LAYER')
        
        dec_token_emb, dec_context_emb, dec_attn = self.self_attn(x, x, x, tgt_mask, one_out=None)
        
        if dec_context_emb is not None and dec_attn is not None: 
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, one_out=True))
            encdec_token_emb, encdec_context_emb, encdec_attn = self.src_attn(x, m, m, src_mask, one_out=None)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, one_out=True))
            return self.sublayer[2](x, self.feed_forward),\
                   dec_token_emb,\
                   dec_context_emb,\
                   dec_attn,\
                   encdec_token_emb,\
                   encdec_context_emb,\
                   encdec_attn

        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, one_out=True))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, one_out=True))
            return self.sublayer[2](x, self.feed_forward),\
                   None, None, None, None, None, None

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class DecMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(DecMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, one_out=None):
        
        global seq_actual_length
        
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        #print('DEC ATTN', self.attn.shape)
        #print('DEC CONTEXT', x.shape)
        #print('DEC EMB', self.linears[-1](x).shape)
        #print(seq_actual_length)
        
        # beam size 5, num heads 8, seq length 17
        # TRAIN EMBEDDINGS: beam size 1, seq length is always the max length of the actual caption
        if self.attn.shape == torch.Size([1, 8, seq_actual_length, seq_actual_length]) and one_out is None:
        #if self.attn.shape == torch.Size([1, 8, seq_actual_length, seq_actual_length]) and one_out is None:
            token_emb = self.linears[-1](x)
            context_emb = x.view(nbatches, -1, self.h, self.d_k)
            context_emb = F.normalize(context_emb, dim=2)
            context_emb = context_emb.view(nbatches, -1, self.h * self.d_k)
            return token_emb, context_emb, self.attn
        if one_out is not None:
            return self.linears[-1](x)
        elif one_out is None and self.attn.shape != torch.Size([1, 8, seq_actual_length, seq_actual_length]):
        #elif one_out is None and self.attn.shape != torch.Size([1, 8, seq_actual_length, seq_actual_length]):
            return self.linears[-1](x), None, None
        
    
class EncDecMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(EncDecMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, one_out=None):
        
        global seq_actual_length
        
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        #print('ENCDEC ATTN', self.attn.shape)
        #print('ENCDEC CONTEXT', x.shape)
        #print('ENCDEC EMB', self.linears[-1](x).shape)
        #print(seq_actual_length)
        
        # beam size 5, num heads 8, seq length 17
        #print(self.attn.shape)
        if self.attn.shape == torch.Size([1, 8, seq_actual_length, 36]) and one_out is None:
        #if self.attn.shape == torch.Size([1, 8, seq_actual_length, 36]) and one_out is None:
            token_emb = self.linears[-1](x)
            context_emb = x.view(nbatches, -1, self.h, self.d_k)
            context_emb = F.normalize(context_emb, dim=2)
            context_emb = context_emb.view(nbatches, -1, self.h * self.d_k)
            return token_emb, context_emb, self.attn
        if one_out is not None:
            return self.linears[-1](x)
        elif one_out is None and self.attn.shape != torch.Size([1, 8, seq_actual_length, 36]):
        #elif one_out is None and self.attn.shape != torch.Size([1, 8, seq_actual_length, 36]):
            return self.linears[-1](x), None, None


def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    #attention weights
    scaled_dot = torch.matmul(w_q,w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn,w_v)

    return output, w_mn

class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding=trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        #matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), 8)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input_query, input_key, input_value, input_box, mask=None, one_out=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = input_query.size(0)

        #tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = utils.BoxRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1,self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        # An extra internal skip connection is added. This is only
        # kept here for compatibility with some legacy models. In
        # general, there is no advantage in using it, as there is
        # already an outer skip connection surrounding this layer.
        if self.legacy_extra_skip:
            x = input_value + x
            
        #print('ENC ATTN', self.box_attn.shape)
        #print('ENC TOKEN EMB', self.linears[-1](x).shape)
        #print('ENC CONTEXT EMB', x.shape)
        
        if self.box_attn.shape == torch.Size([1, 8, 36, 36]) and one_out is None:
            token_emb = self.linears[-1](x)
            context_emb = x.view(nbatches, -1, self.h, self.d_k)
            context_emb = F.normalize(context_emb, dim=2)
            context_emb = context_emb.view(nbatches, -1, self.h * self.d_k)
            return token_emb, context_emb, self.box_attn
        if one_out is not None:
            return self.linears[-1](x)
        elif one_out is None and self.box_attn.shape != torch.Size([1, 8, 36, 36]):
            return self.linears[-1](x), None, None



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 9488 vs 9487 as vocab size
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationTransformerModel(CaptionModel):

    def make_model(self, src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,
                   trignometric_embedding=True, legacy_extra_skip=False):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        bbox_attn = BoxMultiHeadedAttention(h, d_model, trignometric_embedding, legacy_extra_skip)
        dec_attn = DecMultiHeadedAttention(h, d_model)
        enc_dec_attn = EncDecMultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        #position = BoxEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(bbox_attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(dec_attn), c(enc_dec_attn),
                                 c(ff), dropout), N),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(RelationTransformerModel, self).__init__()
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        # #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        # self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        # self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        # self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
        #                         nn.ReLU(),
        #                         nn.Dropout(self.drop_prob_lm))
        # self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
        #                             nn.ReLU(),
        #                             nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))

        self.box_trignometric_embedding = getattr(opt, 'box_trignometric_embedding', True)
        self.legacy_extra_skip = getattr(opt, 'legacy_extra_skip', False)

        tgt_vocab = self.vocab_size + 1
        
        self.model = self.make_model(
            0, tgt_vocab, N=opt.num_layers, d_model=opt.input_encoding_size,
            d_ff=opt.rnn_size,
            trignometric_embedding=self.box_trignometric_embedding,
            legacy_extra_skip=self.legacy_extra_skip)

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters())
    #     return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
    #             weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    # def _prepare_feature(self, fc_feats, att_feats, att_masks):

    #     # embed fc and att feats
    #     fc_feats = self.fc_embed(fc_feats)
    #     att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

    #     # Project the attention feats first to reduce memory and computation comsumptions.
    #     p_att_feats = self.ctx2att(att_feats)

    #     return fc_feats, att_feats, p_att_feats

    def _prepare_feature(self, att_feats, att_masks=None, boxes=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        boxes = self.clip_att(boxes, att_masks)[0]

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            seq_mask[:,0] = 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, boxes, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, boxes,  seq, att_masks=None):

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes, seq)
        out = self.model(att_feats, boxes, seq, att_masks, seq_mask)
        outputs = self.model.generator(out)
        return outputs


    def get_logprobs_state(self, it, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                               ys,
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        logprobs = self.model.generator(out[:, -1])

        return logprobs, [ys.unsqueeze(0)]

    def _sample_beam(self, fc_feats, att_feats, boxes, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = att_feats.size(0)

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)
        memory = self.model.encode(att_feats, boxes, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            tmp_memory = memory[k:k+1].expand(*((beam_size,)+memory.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = att_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_memory, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample_(self, fc_feats, att_feats, boxes, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        if sample_max:
            with torch.no_grad():
                seq_, seqLogprobs_ = self._sample_(fc_feats, att_feats, boxes, att_masks, opt)

        batch_size = att_feats.shape[0]

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)
        memory = self.model.encode(att_feats, boxes, att_masks)
        ys = torch.zeros((batch_size, 1), dtype=torch.long).to(att_feats.device)

        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)

        for i in range(self.seq_length):
            out = self.model.decode(memory, att_masks,
                               ys,
                               subsequent_mask(ys.size(1))
                                        .to(att_feats.device))
            logprob = self.model.generator(out[:, -1])
            if sample_max:
                sampleLogprobs, next_word = torch.max(logprob, dim = 1)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprob.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprob.data, temperature))
                next_word = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, next_word) # gather the logprobs at sampled positions

            seq[:,i] = next_word
            seqLogprobs[:,i] = sampleLogprobs
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        assert (seq*((seq_>0).long())==seq_).all(), 'seq doens\'t match'
        assert (seqLogprobs*((seq_>0).float()) - seqLogprobs_*((seq_>0).float())).abs().max() < 1e-5, 'logprobs doens\'t match'
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, boxes, att_masks=None, opt={}):
                        
        print('ground truth texts', fc_feats, fc_feats.shape)
        global seq_actual_length
        seq_actual_length = torch.count_nonzero(fc_feats).item()
        self.seq_length = seq_actual_length
        
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, boxes, att_masks, opt)

        batch_size = att_feats.shape[0]

        att_feats, boxes, seq, att_masks, seq_mask = self._prepare_feature(att_feats, att_masks, boxes)

        state = None
        memory = self.model.encode(att_feats, boxes, att_masks)

        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)
        
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = att_feats.new_zeros(batch_size, dtype=torch.long)

            #print('this it', it)
            logprobs, state = self.get_logprobs_state(it, memory, att_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                print('actually predicted', it.view(-1).long())
                forced_it = fc_feats[t+1]
                it = forced_it
                it = it.view(-1).long()
                print('forced predicted', it, it.shape)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            #print('final sequence', seq, seq.shape)
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
