# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.bbox.box_coders import GroundBox3dCoderAF


class OHSLossClear(nn.Module):
    """Object as Hotspots Loss Function
    """

    def __init__(self,
                 box_coder: GroundBox3dCoderAF,
                 num_class: int,
                 loss_cls: nn.Module,
                 loss_bbox: nn.Module,
                 encode_background_as_zeros,
                 cfg: Dict,
                 loss_norm: Dict,
                 task_id: Sequence[int],
                 mode: str):
        super().__init__()
        self.cls_out_channels = num_class
        self.pc_range = np.asarray(box_coder.pc_range)
        self.dims = self.pc_range[3:] - self.pc_range[:3]
        self.box_coder = box_coder

        self.loss_cls = loss_cls
        self.loss_box = loss_bbox
        self.loss_norm = loss_norm

        self.encode_background_as_zeros = encode_background_as_zeros

        self.projection = mode

        self.task_id = task_id


    def forward(self,
                bbox_preds: torch.Tensor,
                cls_scores: torch.Tensor,
                labels: torch.Tensor,
                label_weights: torch.Tensor,
                bbox_targets: List[torch.Tensor],
                bbox_locs: List[torch.Tensor],
                **kwargs):
        r"""
        Args:
            bbox_preds (torch.Tensor, [B, H, W, box_dim]): bbox_preds from bev view
            cls_scores (torch.Tensor, [B, H, W, C]): bbox_clssification from bev view
            spatial_relation_preds (List): spatial relation predictions
            labels (torch.Tensorm [B, H*W]): assign labels
            label_weights (torch.Tensorm [B, H*W]): assign label_weights
            bbox_targets (list of torch.Tensor, [[N, box_dim]]): batch list of bounding box regression
            bbox_locs (list of torch.Tensor, [[N, 3]]): batch list of bounding box center indices
        """
        # get the batch size
        bs_per_gpu = len(cls_scores)

        # generate the one hot vector of class labels for calculate cls_losses
        cls_scores = cls_scores.view(bs_per_gpu, -1, self.cls_out_channels)
        one_hot_targets = F.one_hot(labels, self.cls_out_channels + 1).to(cls_scores.dtype)
        if self.encode_background_as_zeros:
            one_hot_targets = one_hot_targets[..., 1:]
        cls_losses = self.loss_cls(cls_scores, one_hot_targets, weights=label_weights)

        # select the positive bounding boxes regression target
        bbox_locs = torch.cat(bbox_locs, 0) # [M, 3]
        bbox_preds = bbox_preds[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :] # [M, box_dim]
        bbox_targets = torch.cat(bbox_targets, 0) # [M, box_dim]
        # ignore the `nan`
        bbox_targets[torch.isnan(bbox_targets)] = bbox_preds[torch.isnan(bbox_targets)].clone().detach()
        loc_losses = self.loss_box(bbox_preds, bbox_targets)

        return loc_losses, cls_losses
