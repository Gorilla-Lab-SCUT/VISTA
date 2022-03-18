# Copyright (c) Gorilla-Lab. All rights reserved.
import logging

import numpy as np
import torch
from torch import nn

from ..registry import DETECTORS
from .voxelnet import VoxelNet
from .. import builder

@DETECTORS.register_module
class OHS_Multiview(VoxelNet):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(OHS_Multiview, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        cfg_rv_neck = dict(
            type=neck['type'],
            layer_nums=[i for i in neck['layer_nums']],
            ds_layer_strides=[i for i in neck['ds_layer_strides']],
            ds_num_filters=[i for i in neck['ds_num_filters']],
            us_layer_strides=[i for i in neck['us_layer_strides']],
            us_num_filters=[i for i in neck['us_num_filters']],
            num_input_features=neck["num_input_features"],
            norm_cfg=neck['norm_cfg'],
            logger=logging.getLogger("RVRPN")
        )
        cfg_ca = dict(
            type='Cross_Attention_Decouple',
            bev_input_channel=384,
            rv_input_channel=384,
            embed_dim=384,
            num_heads=1,
            bev_size=(160, 160),
            bev_block_res=(40, 40),
            rv_size=(160, 12),
            rv_block_res=(40, 12),
            hidden_channels=512,
        )
        self.rv_neck = builder.build_neck(cfg_rv_neck)
        self.ca_neck = builder.build_neck(cfg_ca)
        if train_cfg:
            self.bbox_head.set_train_cfg(train_cfg['assigner'])
        else:
            self.bbox_head.set_train_cfg(test_cfg['assigner'])

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        (x_bev, x_rv) = x
        x_bev = self.neck(x_bev)
        x_rv = self.rv_neck(x_rv)

        return self.ca_neck((x_bev, x_rv))

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        xs, atten_maps = self.extract_feat(data)

        preds = self.bbox_head(xs, return_loss)
        del data['features']
        del example["voxels"]
        # torch.cuda.empty_cache()

        if return_loss:
            return self.bbox_head.loss(example, preds, atten_map=atten_maps)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)