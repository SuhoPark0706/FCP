# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from model.base.MultiHeadAttention import MaskMultiHeadAttention

import math, copy
import numpy as np

def get_gauss(mu, sigma):
    gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, attn_drop_out=0.2,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop_out)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_key, memory_value,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     value_pos: Optional[Tensor] = None):
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt, attn

    def forward_pre(self, tgt, memory_key, memory_value,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    value_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory_key, memory_value,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos)


    def forward(self, tgt, memory_key, memory_value,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos)
        
class CrossAggregationLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.my_attn = MaskMultiHeadAttention(4,256, dropout=0.5)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_key, memory_value,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     value_pos: Optional[Tensor] = None, 
                     add_attn=False):
        
        tgt2, attn = self.my_attn(self.with_pos_embed(tgt, query_pos).permute(1,0,2),
                            self.with_pos_embed(memory_key, pos).permute(1,0,2),
                            self.with_pos_embed(memory_value, value_pos).permute(1,0,2), mask=memory_mask,add_attn=add_attn)
        
        tgt2 = tgt2.permute(1,0,2)
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt, attn

    def forward_pre(self, tgt, memory_key, memory_value,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    value_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory_key, memory_value,
                add_attn=False,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos, add_attn=add_attn)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class transformer_decoder(nn.Module):

    """ Transformer decoder to get point query"""
    def __init__(self, args, num_queries, hidden_dim, dim_feedforward, nheads=4, num_layers=3, pre_norm=False):
        super().__init__()
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.args = args
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = num_layers

        self.transformer_cross_attention_layers_spt = nn.ModuleList()
        self.transformer_cross_attention_layers_qry = nn.ModuleList()
        self.transformer_cross_attention_layers_1 = nn.ModuleList()
        self.transformer_self_attention_layers_1 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):       
            
            self.transformer_cross_attention_layers_spt.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    attn_drop_out = args.attn_drop_out,
                    normalize_before=pre_norm,
                )
            ) 
            self.transformer_cross_attention_layers_qry.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    attn_drop_out = args.attn_drop_out,
                    normalize_before=pre_norm,
                )
            )    

        self.transformer_cross_attention_layers_1.append(
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )

        self.transformer_self_attention_layers_1.append(
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )
        
        self.merge_sam_and_mask = nn.Sequential(    nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
                                                    nn.ReLU(inplace=True)
                                                    )
        self.merge =              nn.Sequential(    nn.Conv2d(hidden_dim*2+1, hidden_dim, kernel_size=1, padding=0, bias=False),
                                                    nn.ReLU(inplace=True)
                                                    )
        
        self.num_queries = num_queries
        self.spt_protos = nn.Embedding(self.args.spt_num_query, hidden_dim)
        self.qry_protos = nn.Embedding(num_queries, hidden_dim)

    def forward(self, x_q, x_s, x_q_sam, x_s_sam, support_mask, spt_prototype, qry_res_feat_4, pseudo_mask, supp_feat_bin, supp_feat_bin_sam):
        bs, C, H, W = x_q.shape
        
        spt_prototype = spt_prototype.squeeze(-1).squeeze(-1).unsqueeze(1)
        
        pos_x_q = self.pe_layer(x_q, None).flatten(2).to(x_q.device).permute(2, 0, 1)
        src_x_q = None
        pos_x_s = self.pe_layer(x_s, None).flatten(2).to(x_s.device).permute(2, 0, 1)
        src_x_s = None
        
        
        src_x_q_sam = x_q_sam.flatten(2).permute(2, 0, 1)

        support_sam_c_attn = []
        output = self.spt_protos.weight.unsqueeze(1).repeat(1, bs, 1)
        for i in range(self.num_layers):
            
            """sam feature updates"""
            x_s_sam_merged = self.merge_sam_and_mask(torch.cat([x_s_sam, supp_feat_bin_sam, support_mask*10], dim=1))
            src_x_s_sam = x_s_sam_merged.flatten(2).permute(2, 0, 1)
            
            if i != self.num_layers-1:
                output, s_c_attn_map = self.transformer_cross_attention_layers_spt[i](
                    output, src_x_s_sam, src_x_s_sam,
                    memory_mask=self.processing_for_attn_mask(support_mask, self.args.spt_num_query),
                    memory_key_padding_mask=None,
                    pos=pos_x_s, query_pos=None
                )
                support_sam_c_attn.append(s_c_attn_map)

            else:      
                x_s = self.merge(torch.cat([x_s, supp_feat_bin, support_mask*10], dim=1))
                src_x_s = x_s.flatten(2).permute(2, 0, 1)
                
                output, s_c_attn_map = self.transformer_cross_attention_layers_spt[i](
                    output, src_x_s_sam, src_x_s,
                    memory_mask=self.processing_for_attn_mask(support_mask, self.args.spt_num_query),
                    memory_key_padding_mask=None,
                    pos=pos_x_s, query_pos=None
                )
                support_sam_c_attn.append(s_c_attn_map)
                
        spt_protos = output
        
        pseudo_mask_loss = []
        query_sam_c_attn = []
        pseudo_mask_vis = []

        pseudo_mask_vis.append(pseudo_mask.float())
        pseudo_mask_vis.append((pseudo_mask>0.5).float())

        output = self.qry_protos.weight.unsqueeze(1).repeat(1, bs, 1)
        for i in range(self.num_layers):
            
            pseudo_mask_naive = pseudo_mask
            
            """sam feature updates"""
            if self.args.concat_th:
                pseudo_mask_for_concat = (pseudo_mask>0.5).float()
            else:
                pseudo_mask_for_concat = pseudo_mask
                
            x_q_sam_merged = self.merge_sam_and_mask(torch.cat([x_q_sam, supp_feat_bin_sam, pseudo_mask_naive*10], dim=1))
            src_x_q_sam = x_q_sam_merged.flatten(2).permute(2, 0, 1)
  
            if i != self.num_layers-1:
                output, q_c_attn_map = self.transformer_cross_attention_layers_qry[i](
                    output, src_x_q_sam, src_x_q_sam,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=pos_x_q, query_pos=None, value_pos=None
                )
                query_sam_c_attn.append(q_c_attn_map)

                """pseudo mask"""
                # min max norm
                q_c_attn_map = ( q_c_attn_map - q_c_attn_map.min(-1, keepdim=True)[0] ) / ( q_c_attn_map.max(-1, keepdim=True)[0] - q_c_attn_map.min(-1, keepdim=True)[0] + 1e-9 )

                # mask merge
                cur_pseudo_mask = q_c_attn_map.max(1, keepdim=True)[0]
                pseudo_mask = cur_pseudo_mask.reshape(bs,1,64,64)

                pseudo_mask_for_loss = pseudo_mask                
                pseudo_mask_loss.append( pseudo_mask_for_loss )
                pseudo_mask_vis.append( pseudo_mask )
                pseudo_mask_vis.append( (pseudo_mask>0.5).float() )
                
            else:
                """sam feature updates"""
                if self.args.concat_th:
                    pseudo_mask_for_concat = (pseudo_mask>0.5).float()
                else:
                    pseudo_mask_for_concat = pseudo_mask
                    
                x_q = self.merge(torch.cat([x_q, supp_feat_bin, pseudo_mask_for_concat*10], dim=1))
                src_x_q = x_q.flatten(2).permute(2, 0, 1)
                                
                output, q_c_attn_map = self.transformer_cross_attention_layers_qry[i](
                    output, src_x_q_sam, src_x_q,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=pos_x_q, query_pos=None, value_pos=None
                )
                query_sam_c_attn.append(q_c_attn_map)
                
        qry_protos = output

        for i in range(1):
            output, _ = self.transformer_cross_attention_layers_1[i](
                spt_protos, qry_protos, qry_protos,
                memory_mask=None,
                memory_key_padding_mask=None,  
                pos=None, query_pos=None, value_pos=None
            )
            output, _ = self.transformer_self_attention_layers_1[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

        return output.permute(1, 0, 2), [s_c_attn_map, q_c_attn_map, pseudo_mask_vis, (q_c_attn_map).float()], \
                {   'pseudo_mask_loss': pseudo_mask_loss, \
                    'query_sam_c_attn' : query_sam_c_attn, 'support_sam_c_attn' : support_sam_c_attn}

    def processing_for_attn_mask(self, mask, num, empty_check=False):
        mask = mask.flatten(2)
        # check empty pseudo mask
        if empty_check:
            empty_mask = (mask.sum(-1, keepdim=True) == 0.).float()
            mask = mask + empty_mask * 1.0
        
        # arrange
        mask = mask.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.num_heads, num, 1).flatten(start_dim=0, end_dim=2)
        mask = mask == 0.
        return mask