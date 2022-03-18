# Copyright (c) Gorilla-Lab. All rights reserved.

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..registry import NECKS


class ConvAttentionLayer_Decouple(nn.Module):
    def __init__(self, input_channels: int, numhead: int = 1, reduction_ratio=2):
        r"""
        Args:
            input_channels (int): input channel of conv attention
            numhead (int, optional): the number of attention heads. Defaults to 1.
        """
        super().__init__()
        # self.q_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.q_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        self.q_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        # self.k_conv = nn.Conv2d(input_channels, input_channels // reduction_ratio, 3, 1, 1)
        self.k_sem_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        self.k_geo_conv = nn.Conv2d(input_channels,
                                    input_channels // reduction_ratio, 3, 1, 1)
        # self.v_conv = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.v_conv = nn.Conv2d(input_channels, input_channels, 1, 1, 0)
        self.out_sem_conv = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.out_geo_conv = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.channels = input_channels // reduction_ratio
        self.numhead = numhead
        self.head_dim = self.channels // numhead
        self.sem_norm = nn.LayerNorm(input_channels)
        self.geo_norm = nn.LayerNorm(input_channels)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                q_pos_emb: Optional[torch.Tensor] = None,
                k_pos_emb: Optional[torch.Tensor] = None):
        r"""
        Args:
            query (torch.Tensor, [B, C, H_qk, W_qk]): feature of query
            key (torch.Tensor, [B, C, H_qk, W_qk]): feature of key
            value (torch.Tensor, [B, C, H_v, W_v]): feature of value
            q_pos_emb (Optional[torch.Tensor], optional, [[B, C, H_q, W_q]]):
                positional encoding. Defaults to None.
            k_pos_emb (Optional[torch.Tensor], optional, [[B, C, H_kv, W_kv]]):
                positional encoding. Defaults to None.
        """

        view = query + 0  # NOTE: a funny method to deepcopy
        input_channel = view.shape[1]
        if q_pos_emb is not None:
            query += q_pos_emb
        if k_pos_emb is not None:
            key += k_pos_emb

        # to qkv forward
        # q = self.q_conv(query)
        q_sem = self.q_sem_conv(query)
        q_geo = self.q_geo_conv(query)
        qs = [q_sem, q_geo]
        # k = self.k_conv(key)
        k_sem = self.k_sem_conv(key)
        k_geo = self.k_geo_conv(key)
        ks = [k_sem, k_geo]
        v = self.v_conv(value)
        vs = [v, v]
        out_convs = [self.out_sem_conv, self.out_geo_conv]
        norms = [self.sem_norm, self.geo_norm]
        outputs = []
        attentions = []

        for q, k, v, out_conv, norm in zip(qs, ks, vs, out_convs, norms):
            # read shape of qkv
            bs = q.shape[0]
            qk_channel = q.shape[1]  # equal to the channel of `k`
            v_channel = v.shape[1]  # channel of `v`
            h_q, w_q = q.shape[2:]  # height and weight of query map
            h_kv, w_kv = k.shape[2:]  # height and weight of key and value map
            numhead = self.numhead
            qk_head_dim = qk_channel // numhead
            v_head_dim = v_channel // numhead

            # scale query
            scaling = float(self.head_dim) ** -0.5
            q = q * scaling

            # reshape(sequentialize) qkv
            q = rearrange(q, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_q, w=w_q)
            q = rearrange(q, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                          n=numhead, h=h_q, w=w_q, d=qk_head_dim)
            q = q.contiguous()
            k = rearrange(k, "b c h w -> b c (h w)", b=bs, c=qk_channel, h=h_kv, w=w_kv)
            k = rearrange(k, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                          n=numhead, h=h_kv, w=w_kv, d=qk_head_dim)
            k = k.contiguous()
            v = rearrange(v, "b c h w -> b c (h w)", b=bs, c=v_channel, h=h_kv, w=w_kv)
            v = rearrange(v, "b (n d) (h w) -> (b n) (h w) d", b=bs,
                          n=numhead, h=h_kv, w=w_kv, d=v_head_dim)
            v = v.contiguous()

            # get the attention map
            energy = torch.bmm(q, k.transpose(1, 2))  # [h_q*w_q, h_kv*w_kv]
            attention = F.softmax(energy, dim=-1)  # [h_q*w_q, h_kv*w_kv]
            attentions.append(attention)
            # get the attention output
            r = torch.bmm(attention, v)  # [bs * nhead, h_q*w_q, C']
            r = rearrange(r, "(b n) (h w) d -> b (n d) h w", b=bs,
                          n=numhead, h=h_q, w=w_q, d=v_head_dim)
            r = r.contiguous()
            r = out_conv(r)

            # residual
            temp_view = view + r
            temp_view = temp_view.view(bs, input_channel, -1).permute(2, 0, 1).contiguous()
            temp_view = norm(temp_view)
            outputs.append(temp_view)
        return outputs, attentions


class FeedFowardLayer(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, input_channel)

        self.norm = nn.LayerNorm(input_channel)

        self.activation = nn.ReLU()  # con be modify as GeLU or GLU

    def forward(self, view: torch.Tensor):
        ffn = self.linear2(self.activation((self.linear1(view))))
        view = view + ffn
        view = self.norm(view)
        return view


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, res: Sequence[int], num_pos_feats=384):
        r"""
        Absolute pos embedding, learned.
        Args:
            res (Sequence[int]): resolution (height and width)
            num_pos_feats (int, optional): the number of feature channel. Defaults to 384.
        """
        super().__init__()
        h, w = res
        num_pos_feats = num_pos_feats // 2
        self.row_embed = nn.Embedding(h, num_pos_feats)
        self.col_embed = nn.Embedding(w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).contiguous().unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class CrossAttenBlock_Decouple(nn.Module):
    def __init__(self,
                 input_channels: int,
                 numhead: int = 1,
                 hidden_channel: int = 2048,
                 reduction_ratio=2):
        r"""
        Block of cross attention (one cross attention layer and one ffn)
        Args:
            input_channels (int): input channel of conv attention
            numhead (int, optional): the number of attention heads. Defaults to 1.
            hidden_channel (int, optional): channel of ffn. Defaults to 2048.
        """
        super().__init__()
        self.cross_atten = ConvAttentionLayer_Decouple(input_channels, numhead, reduction_ratio)
        self.ffn_sem = FeedFowardLayer(input_channels, hidden_channel)
        self.ffn_geo = FeedFowardLayer(input_channels, hidden_channel)

    def forward(self,
                view_1: torch.Tensor,
                view_2: torch.Tensor,
                pos_emb_1: Optional[torch.Tensor] = None,
                pos_emb_2: Optional[torch.Tensor] = None):
        B, C, H, W = view_1.shape
        views, atten_maps= self.cross_atten(view_1, view_2, view_2, pos_emb_1, pos_emb_2)
        ffns = [self.ffn_sem, self.ffn_geo]
        outputs = []
        for i in range(len(views)):
            ffn = ffns[i]
            view = ffn(views[i])
            view = view.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()
            outputs.append(view)
        return outputs, atten_maps

@NECKS.register_module
class Cross_Attention_Decouple(nn.Module):
    def __init__(self,
                 bev_input_channel: int,
                 rv_input_channel: int,
                 embed_dim: int,
                 num_heads: int,
                 bev_size: Sequence[int] = (262, 64),
                 bev_block_res: Sequence[int] = (16, 16),
                 rv_size: Sequence[int] = (262, 64),
                 rv_block_res: Sequence[int] = (16, 16),
                 hidden_channels: Sequence[int] = 1024):
        r"""
        Convolutional Cross-View Transformer module
        Args:
            bev_input_channel (int): input channels of bev feature
            rv_input_channel (int): input channels of rv feature
            embed_dim (int): channels of downsample(input for attention)
            num_heads (int): the number of attention heads
            num_conv (int, optional): the number of convolutional layers. Defaults to 5.
            bev_size (Sequence[int], optional): size of bev feature map. Defaults to (262, 64).
            bev_block_res (Sequence[int], optional): size of rv feature map. Defaults to (16, 16).
            rv_size (Sequence[int], optional): size of bev feature map. Defaults to (262, 64).
            rv_block_res (Sequence[int], optional): size of rv feature map. Defaults to (16, 16).
            hidden_channels (Sequence[int], optional): channels of ffn. Defaults to [512, 1024].
            block_feat_mode (str, optional): the manner to obtain the block-wise feature
                                             "AVERAGE" : the block feature will be the average feature inside the block
                                             "CONV" : use conv layer to obtain the block feature. Defaults to "AVERAGE".
        """
        super().__init__()
        assert len(bev_block_res) == 2
        self.bev_input_channel = bev_input_channel
        self.rv_input_channel = rv_input_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bev_size = bev_size
        self.rv_size = rv_size
        self.bev_block_res = self.adjust_res(bev_block_res, bev_size)
        self.rv_block_res = self.adjust_res(rv_block_res, rv_size)

        # positional encoding
        self.bev_pos_emb = PositionEmbeddingLearned(self.bev_block_res, embed_dim)
        self.rv_pos_emb = PositionEmbeddingLearned(self.rv_block_res, embed_dim)

        self.cross_atten = CrossAttenBlock_Decouple(embed_dim, num_heads, hidden_channels)

    def adjust_res(self, block_res: Sequence[int], size: Sequence[int]):
        h, w = block_res[0], block_res[1]
        H, W = size[0], size[1]
        assert H % h == 0, f"H must be divisible by h, but got H-{H}, h-{h}"
        assert W % w == 0, f"H must be divisible by h, but got H-{W}, h-{w}"
        h_step = math.ceil(H / h)
        w_step = math.ceil(W / w)
        h_length = math.ceil(H / h_step)
        w_length = math.ceil(W / w_step)
        return (h_length, w_length)

    def generate_feat_block(self, feat_map: torch.Tensor, block_res: Sequence[int]):
        R"""
        generate feat block and the size of each block by scatter_mean
        Args:
            feat_map (torch.Tensor, [B, C, H, W]): input feature map
            block_res  (Sequence[int]): (block_x, block_y)
        Returns:
            block_feat_map (torch.Tensor, [B, C, x_block, y_block]): feature map of blocks
            kernel_size: (Tuple[int]) avg pooling kernel size
        """
        H, W = feat_map.shape[-2:]
        kernel_size = (int(H / block_res[0]), int(W / block_res[1]))
        block_feat_map = F.avg_pool2d(feat_map, kernel_size, kernel_size)

        return block_feat_map, kernel_size

    def residual_add(self,
                     feat_map: torch.Tensor,
                     attn_block: torch.Tensor,
                     kernel_size: Sequence[int]) -> torch.Tensor:
        r"""
        residual realization briefly
        Args:
            feat_map (torch.Tensor, [B, D, H, W]): origin feature map
            attn_block (torch.Tensor, [B, D, h_block, w_block]): attention block feature map
            feat_num_block (torch.Tensor, [h_block, w_block, 2]): size storage of each block
        Returns:
            torch.Tensor, [B, D, H, W]: origin feature map add attention back prjection feature map
        """
        h_backproj_feature_new = torch.repeat_interleave(attn_block, kernel_size[0], dim=2)
        hw_backproj_feature = torch.repeat_interleave(h_backproj_feature_new, kernel_size[1], dim=3)
        # output = torch.cat([feat_map, hw_backproj_feature], 1)
        output = feat_map + hw_backproj_feature

        return output

    def forward(self, x: Sequence[torch.Tensor]):
        r"""
        Args:
            x (Sequence[torch.Tensor]):
                bev_feat_map_up (torch.Tensor, [B, embed_dim, H1, W1]), feature map of bev
                rvv_feat_map_up (torch.Tensor, [B, embed_dim, H2, W2]), feature map of rv
        """
        (bev_feat_map, rv_feat_map) = x

        assert (*bev_feat_map.shape[-2:],) == tuple(self.bev_size), (f"get the size of bev feature map - {bev_feat_map.shape[-2:]}, "
                                                                     f"which does not match the given size {self.bev_size}")
        assert (*rv_feat_map.shape[-2:],) == tuple(self.rv_size), (f"get the size of rv feature map - {rv_feat_map.shape[-2:]}, "
                                                                   f"which does not match the given size {self.rv_size}")

        # generate feature block of bev (for attention)
        bev_feat_block, bev_kernel_size = self.generate_feat_block(
            bev_feat_map, self.bev_block_res)

        # generate feature block of rv (for attention)
        rv_feat_block, rv_kernel_size = self.generate_feat_block(
            rv_feat_map, self.rv_block_res)

        # generate positional encoding
        bev_pos = self.bev_pos_emb(bev_feat_block)
        rv_pos = self.rv_pos_emb(rv_feat_block)
        bev_output_feat_maps = []

        bev_atten_outputs, atten_maps= self.cross_atten(
            bev_feat_block, rv_feat_block, bev_pos, rv_pos)
        for bev_atten_output in bev_atten_outputs:
            bev_output_feat_map = self.residual_add(bev_feat_map,
                                                    bev_atten_output,
                                                    bev_kernel_size).contiguous()
            bev_output_feat_maps.append(bev_output_feat_map)
        return bev_output_feat_maps, atten_maps
