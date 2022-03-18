# Copyright (c) Gorilla-Lab. All rights reserved.
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS
from ..builder import build_loss
from ...core.bbox.box_coders import BoxCoder, GroundBox3dCoderAF
from .clear_mg_ohs_head import MultiGroupOHSHeadClear


@HEADS.register_module
class DeepOHSHeadClear_Decouple(nn.Module):
    def __init__(self,
                 box_coder: GroundBox3dCoderAF,
                 num_input: int,
                 num_pred: int,
                 num_cls: int,
                 header: bool = True,
                 name: str = "",
                 **kwargs):
        super().__init__()

        self.conv_head = num_input

        self.box_coder = box_coder
        self.conv_cls = nn.Sequential(
            nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
            nn.BatchNorm2d(self.conv_head),
            nn.ReLU(),
            nn.Conv2d(self.conv_head, num_cls, 3, 1, 1)
        )
        self.mode = kwargs.get("mode", "bev")
        if self.box_coder.center == "direct":
            self.conv_xy = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 2, 3, 1, 1)
            )
        elif self.box_coder.center == "soft_argmin":
            self.conv_xy = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 2 * self.box_coder.kwargs["xy_bin_num"], 3, 1, 1)
            )
            self.loc_bins_x = torch.linspace(self.box_coder.kwargs["x_range"][0], self.box_coder.kwargs["x_range"][1],
                                             self.box_coder.kwargs["xy_bin_num"]).reshape(1, 1, -1, 1, 1)
            self.loc_bins_y = torch.linspace(self.box_coder.kwargs["y_range"][0], self.box_coder.kwargs["y_range"][1],
                                             self.box_coder.kwargs["xy_bin_num"]).reshape(1, 1, -1, 1, 1)
            self.loc_bins = torch.cat([self.loc_bins_x, self.loc_bins_y], 1).cuda()
        else:
            raise NotImplementedError

        if "direct" in self.box_coder.height:
            self.conv_z = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 1, 3, 1, 1)
            )
        elif "soft_argmin" in self.box_coder.height:
            self.conv_z = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, self.box_coder.kwargs["z_bin_num"], 3, 1, 1)
            )
            self.z_loc_bins = torch.linspace(self.box_coder.kwargs["z_range"][0], self.box_coder.kwargs["z_range"][1],
                                             self.box_coder.kwargs["z_bin_num"]).reshape(1, self.box_coder.kwargs["z_bin_num"], 1, 1).cuda()
        else:
            raise NotImplementedError
        if "soft_argmin" in self.box_coder.dim:
            self.conv_dim = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 3 * self.box_coder.kwargs["dim_bin_num"], 3, 1, 1)
            )
            self.dim_loc_bins = torch.linspace(self.box_coder.kwargs["dim_range"][0], self.box_coder.kwargs["dim_range"][1],
                                               self.box_coder.kwargs["dim_bin_num"]).reshape(1, self.box_coder.kwargs[
                                                   "dim_bin_num"], 1, 1).cuda()
            self.dim_bins = torch.cat([self.dim_loc_bins, self.dim_loc_bins, self.dim_loc_bins], 1)
        else:
            self.conv_dim = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 3, 3, 1, 1)
            )
        if self.box_coder.velocity:
            self.conv_velo = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 2, 3, 1, 1)
            )
        if self.box_coder.rotation == "vector":
            self.conv_r = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 2, 3, 1, 1)
            )
        elif self.box_coder.rotation == "soft_argmin":
            self.conv_r = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, self.box_coder.kwargs["r_bin_num"], 3, 1, 1)
            )
            self.r_loc_bins = torch.linspace(-np.pi, np.pi, self.box_coder.kwargs["r_bin_num"]).reshape(
                1, self.box_coder.kwargs["r_bin_num"], 1, 1).cuda()
        else:
            self.conv_r = nn.Sequential(
                nn.Conv2d(num_input, self.conv_head, 3, 1, 1),
                nn.BatchNorm2d(self.conv_head),
                nn.ReLU(),
                nn.Conv2d(self.conv_head, 1, 3, 1, 1)
            )

    def forward(self, xs, return_loss):
        # read the feature map for different view and perform classification prediction
        x_sem, x_geo = xs
        ret_dict = {}
        cls_preds = self.conv_cls(x_sem).permute(0, 2, 3, 1).contiguous()

        # predict bounding box
        xy = self.conv_xy(x_geo)
        z = self.conv_z(x_geo)
        dim = self.conv_dim(x_geo)
        # encode as bounding box
        if self.box_coder.center == "soft_argmin":
            xy = xy.view(
                (xy.shape[0], 2, self.box_coder.kwargs["xy_bin_num"], xy.shape[2], xy.shape[3]))
            xy = F.softmax(xy, dim=2)
            xy = xy * self.loc_bins.to(xy.device)
            xy = torch.sum(xy, dim=2, keepdim=False)
        if "soft_argmin" in self.box_coder.height:
            z = F.softmax(z, dim=1)
            z = z * self.z_loc_bins.to(z.device)
            z = torch.sum(z, dim=1, keepdim=True)
        if "soft_argmin" in self.box_coder.dim:
            dim = dim.view(
                (dim.shape[0], 3, self.box_coder.kwargs["dim_bin_num"], dim.shape[2], dim.shape[3]))
            dim = F.softmax(dim, dim=2)
            dim = dim * self.dim_loc_bins.to(dim.device)
            dim = torch.sum(dim, dim=2, keepdim=False)
        xy = xy.permute(0, 2, 3, 1).contiguous()
        z = z.permute(0, 2, 3, 1).contiguous()
        dim = dim.permute(0, 2, 3, 1).contiguous()

        if self.box_coder.dim == "direct":
            dim = F.relu(dim)

        if self.box_coder.velocity:
            velo = self.conv_velo(x_geo).permute(0, 2, 3, 1).contiguous()

        r_preds = self.conv_r(x_geo)
        if self.box_coder.rotation == "vector":
            #import pdb; pdb.set_trace()
            r_preds = F.normalize(r_preds, p=2, dim=1)
        elif self.box_coder.rotation == "soft_argmin":
            r_preds = F.softmax(r_preds, dim=1)
            r_preds = r_preds * self.r_loc_bins.to(r_preds.device)
            r_preds = torch.sum(r_preds, dim=1, keepdim=True)
        r_preds = r_preds.permute(0, 2, 3, 1).contiguous()
        if self.box_coder.velocity:
            box_preds = torch.cat([xy, z, dim, velo, r_preds], -1)
        else:
            box_preds = torch.cat([xy, z, dim, r_preds], -1)

        ret_dict.update({"box_preds": box_preds, "cls_preds": cls_preds})

        return ret_dict


@HEADS.register_module
class DeepMultiGroupOHSHeadClear_Decouple(MultiGroupOHSHeadClear):
    def __init__(self, mode: str = "3d",
                 in_channels: List[int] = [128, ],
                 norm_cfg=None,
                 tasks: List[Dict] = [],
                 weights=[],
                 box_coder: BoxCoder = None,
                 with_cls: bool = True,
                 with_reg: bool = True,
                 encode_background_as_zeros: bool = True,
                 use_sigmoid_score: bool = True,
                 loss_norm: Dict = dict(type="NormByNumPositives",
                                        pos_class_weight=1.0,
                                        neg_class_weight=1.0,),
                 loss_cls: Dict = dict(type="CrossEntropyLoss",
                                       use_sigmoid=False,
                                       loss_weight=1.0,),
                 loss_bbox: Dict = dict(type="SmoothL1Loss",
                                        beta=1.0,
                                        loss_weight=1.0,),
                 atten_res: Sequence[int] = None,
                 assign_cfg: Optional = dict(),
                 name="rpn"):
        super().__init__(mode=mode, in_channels=in_channels, norm_cfg=norm_cfg, tasks=tasks, weights=weights, box_coder=box_coder, with_cls=with_cls, with_reg=with_reg, encode_background_as_zeros=encode_background_as_zeros, use_sigmoid_score=use_sigmoid_score,
                         loss_norm=loss_norm, loss_cls=loss_cls, loss_bbox=loss_bbox, atten_res=atten_res, assign_cfg=assign_cfg, name=name)
        assert with_cls or with_reg

        # read tasks and analysis the classes for tasks
        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.num_anchor_per_locs = [1] * len(num_classes)
        self.targets = tasks

        # define the essential paramters
        self.box_coder = box_coder
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_sigmoid_score = use_sigmoid_score
        self.box_n_dim = self.box_coder.n_dim
        self.mode = mode
        self.assign_cfg = assign_cfg
        self.pc_range = np.asarray(self.box_coder.pc_range)  # [6]
        self.dims = self.pc_range[3:] - self.pc_range[:3]  # [3]

        self.loss_norm = loss_norm
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_bbox)
        self.atten_res = atten_res

        # initialize logger
        logger = logging.getLogger("MultiGroupHead_Decouple")
        self.logger = logger

        # check box_coder
        assert isinstance(
            box_coder, GroundBox3dCoderAF), "OHSLoss must comes with an anchor-free box coder"
        assert box_coder.code_size == len(
            loss_bbox.code_weights), "code weights does not match code size"

        # set multi-tasks heads
        # split each head
        self.conv_head = 64
        self.sem_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.conv_head, 3, 1, 1),
            nn.BatchNorm2d(self.conv_head),
            nn.ReLU()
        )
        self.geo_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.conv_head, 3, 1, 1),
            nn.BatchNorm2d(self.conv_head),
            nn.ReLU()
        )

        num_clss = []
        num_preds = []
        box_code_sizes = [self.box_coder.n_dim] * len(self.num_classes)
        for num_c, num_a, box_cs in zip(
                self.num_classes, self.num_anchor_per_locs, box_code_sizes
        ):
            if self.encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)

            num_pred = num_a * box_cs
            num_preds.append(num_pred)

        self.logger.info(
            f"num_classes: {self.num_classes}, num_preds: {num_preds}"
        )

        # construct each task head
        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(
                DeepOHSHeadClear_Decouple(
                    self.box_coder,
                    self.conv_head,
                    num_pred,
                    num_cls,
                    header=False,
                    mode=self.mode,
                )
            )

    def forward(self, xs, return_loss=False):
        ret_dicts = []
        x_sem, x_geo = xs
        x_sem = self.sem_conv(x_sem)
        x_geo = self.geo_conv(x_geo)
        for task in self.tasks:
            ret_dicts.append(task((x_sem, x_geo), return_loss))

        return ret_dicts
