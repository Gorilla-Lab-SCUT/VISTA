from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import time
import pdb
import torch
from torch import nn
import spconv
import logging
from .. import builder
import pickle
import os
import numpy as np


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
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
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

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
        x = self.extract_feat(data)
        preds = self.bbox_head(x)
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)


@DETECTORS.register_module
class VoxelNetOHS(VoxelNet):
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
        if bbox_head.mode in ['rv2bev', 'rv_bev', 'bev2rv', 'cycle']:
            grid_size = reader.pop('grid_size')
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        if train_cfg:
            self.bbox_head.set_train_cfg(train_cfg['assigner'])
            self.occupancy = train_cfg['assigner'].get('occupancy', False)
        else:
            self.bbox_head.set_train_cfg(test_cfg['assigner'])
        if not (self.bbox_head.mode in ['bev', 'rv']):
            if 'FHD' in self.backbone.name:
                ds_factor = 16
            else:
                ds_factor = 8
            if grid_size[0] % ds_factor:
                num_input_features = (grid_size[0] // ds_factor + 1) * 128
            else:
                num_input_features = (grid_size[0] // ds_factor) * 128

            # cfg_rv_neck = dict(
            #     type=neck['type'],
            #     layer_nums=[5, 5],
            #     ds_layer_strides = [1, 2],
            #     ds_num_filters=[64, 128],
            #     us_layer_strides=[1, 2],
            #     us_num_filters=[128, 64],
            #     num_input_features=(grid_size[0]//8+1)*128,
            #     norm_cfg=neck['norm_cfg'],
            #     logger=logging.getLogger("RVRPN")
            # )
            # import pdb; pdb.set_trace()
            cfg_rv_neck = dict(
                type=neck['type'],
                layer_nums=[i for i in neck['layer_nums']],
                ds_layer_strides=[i for i in neck['ds_layer_strides']],
                ds_num_filters=[i // 2 for i in neck['ds_num_filters']],
                us_layer_strides=[i for i in neck['us_layer_strides']],
                us_num_filters=[i // 2 for i in neck['us_num_filters']],
                num_input_features=num_input_features,
                norm_cfg=neck['norm_cfg'],
                logger=logging.getLogger("RVRPN")
            )

            self.rv_neck = builder.build_neck(cfg_rv_neck)

            if backbone.mode in ['rv2bev', 'cycle']:
                if grid_size[0] % 8:
                    self.rv2bev = nn.Conv2d(
                        grid_size[-1] // 8, grid_size[0] // 8 + 1, kernel_size=1)
                    #self.rv2bev = nn.MaxPool2d(grid_size[-1] // 8)
                    #self.rv2bev = nn.AvgPool2d(grid_size[-1] // 8)
                else:
                    self.rv2bev = nn.Conv2d(grid_size[-1] // 8, grid_size[0] // 8, kernel_size=1)
                #self.fuse_layer = nn.Conv2d(grid_size[-1] // 8, grid_size[0] // 8 + 1, kernel_size=1)
            if backbone.mode in ['bev2rv', 'cycle']:
                if grid_size[0] % 8:
                    self.bev2rv = nn.Conv2d(grid_size[0] // 8 + 1,
                                            grid_size[-1] // 8, kernel_size=1)
                else:
                    self.bev2rv = nn.Conv2d(grid_size[0] // 8, grid_size[-1] // 8, kernel_size=1)
            # self.rv_layers = nn.Sequential(
            # nn.Conv2d(8192, 8192//4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8192//4),
            # nn.ReLU(),
            # nn.Conv2d(8192//4, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            # nn.ReLU(),
            # )

    def extract_feat(self, data, return_loss):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if (self.bbox_head.mode in ['bev', 'rv']) and self.with_neck:
            return self.neck(x)
        else:
            x_bev, x_rv = x
            # N, C, D, H, W = x.shape
            # x_bev = self.neck(x.view(N, C * D, H, W))
            # x = x.permute(0,1,4,3,2)
            # x_rv = self.rv_layers(x.contiguous().view(N, C * W, H, D))
            x_bev = self.neck(x_bev)
            if return_loss or self.backbone.mode in ['rv2bev', 'cycle']:
                x_rv = self.rv_neck(x_rv)
                return (x_bev, x_rv)
            return x_bev

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
        x = self.extract_feat(data, return_loss)
        grid_size = example['shape'][0]
        preds = self.bbox_head(x, return_loss)
        del data['features']
        del example["voxels"]
        # torch.cuda.empty_cache()

        if self.bbox_head.mode == 'rv2bev':
            for i in range(len(preds)):
                if return_loss:
                    preds[i]['combine_score'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
                else:
                    # for k in range(batch_size):
                    #     shabi = torch.sigmoid(preds[i]['cls_preds'][k]).cpu().numpy()
                    #     np.save(os.path.join('bev_map3', example['metadata'][k]['token']) + '_%i' % i, shabi)
                    preds[i]['cls_preds'] = 0.2 * self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3))\
                        .permute(0, 2, 1, 3) + 0.8 * preds[i]['cls_preds']
                    # if using max/avg pooling
                    # preds[i]['cls_preds'] = 0.2 * self.rv2bev(preds[i]['rv_cls_preds']).repeat(1,1,preds[i]['cls_preds'].shape[2],1) \
                    #      + 0.8 * preds[i]['cls_preds']

                    # preds[i]['cls_preds'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                    #     .permute(0, 2, 1, 3).contiguous()
                    # preds[i]['cls_preds'] = 0.2 * self.fuse_layer(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                    #      .permute(0, 2, 1, 3) + 0.8 * preds[i]['cls_preds']
        if self.bbox_head.mode == 'bev2rv':
            for i in range(len(preds)):
                if return_loss:
                    preds[i]['combine_score'] = self.bev2rv(preds[i]['cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
        if self.bbox_head.mode == 'cycle':
            for i in range(len(preds)):
                if return_loss:
                    preds[i]['combine_score'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
                    preds[i]['bev2rv_score'] = self.bev2rv(preds[i]['cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
                else:
                    # pass
                    preds[i]['cls_preds'] = 0.2 * self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3))\
                        .permute(0, 2, 1, 3) + 0.8 * preds[i]['cls_preds']
                    #preds[i]['cls_preds'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3))
                    # preds[i]['combine_score'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                    #     .permute(0, 2, 1, 3)
                    # preds[i]['bev2rv_score'] = self.bev2rv(preds[i]['cls_preds'].permute(0, 2, 1, 3)) \
                    #     .permute(0, 2, 1, 3)
                # for k in range(batch_size):
                #     shabi = torch.sigmoid(preds[i]['cls_preds'][k]).cpu().numpy()
                #     np.save(os.path.join('bev_map',example['metadata'][k]['token'])+'_%i'%i, shabi)
                #
                #     shabi = torch.sigmoid(preds[i]['combine_score'][k]).cpu().numpy()
                #     np.save(os.path.join('rv_map', example['metadata'][k]['token']) + '_%i'%i, shabi)
                #
                    # shabi = torch.sigmoid(0.2 * preds[i]['combine_score'][k] + 0.8 * preds[i]['cls_preds'][k]).cpu().numpy()
                    # np.save(os.path.join('cycle_map', example['metadata'][k]['token']) + '_%i'%i, shabi)
        if return_loss:
            if self.occupancy:
                occupancy = spconv.SparseConvTensor(
                    torch.ones((len(coordinates), 1), device=coordinates.device,
                               dtype=x.dtype), coordinates.int(),
                    grid_size[::-1], batch_size).dense().squeeze(1)
                occupancy = nn.AdaptiveMaxPool2d(x.shape[-2:])(occupancy).detach()
                occupancy, _ = torch.max(occupancy, dim=1)
                occupancy = occupancy.bool()
            else:
                occupancy = None
            return self.bbox_head.loss(example, preds, occupancy=None)
        else:
            # self.bbox_head.loss(example, preds, occupancy=None)

            # print(self.bbox_head.ohs_loss[0].shabi_box[0])
            # print(self.bbox_head.ohs_loss[0].shabi_loc[0])
            # # for k in range(batch_size):
            # #     for i in range(len(self.bbox_head.ohs_loss)):
            # #         shabi = ((self.bbox_head.ohs_loss[i].shabi_score.view(preds[i]['cls_preds'].shape))[k]).cpu().numpy()
            # #         np.save(os.path.join('gt_map', example['metadata'][k]['token']) + '_%i'%i, shabi)
            # for i in range(len(self.bbox_head.ohs_loss)):
            #     preds[i]['cls_preds'] = self.bbox_head.ohs_loss[i].shabi_score.view(preds[i]['cls_preds'].shape)
            #     preds[i]['box_preds'][self.bbox_head.ohs_loss[i].shabi_loc[:, 0], self.bbox_head.ohs_loss[i].shabi_loc[:,1], self.bbox_head.ohs_loss[i].shabi_loc[:,2], -2:]=self.bbox_head.ohs_loss[i].shabi_box[:,-2:]
            #     #assert (preds[i]['box_preds'][self.bbox_head.ohs_loss[i].shabi_loc[:, 0], self.bbox_head.ohs_loss[i].shabi_loc[:,1], self.bbox_head.ohs_loss[i].shabi_loc[:,2],3:6]>0).all() , "size should be larger than 0"
            # gt_len = 0
            # for i in example['annos'][0]['gt_boxes']:
            #     gt_len += len(i)
            # assert (len(shabi[0]['box3d_lidar']) <= gt_len)
            return self.bbox_head.predict(example, preds, self.test_cfg)


@DETECTORS.register_module
class VoxelNetCA(VoxelNet):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        save_attention=False,
        rv2bev_save_attention_path=None,
        bev2rv_save_attention_path=None

    ):
        if bbox_head.mode in ['rv2bev', 'rv_bev', 'bev2rv', 'cycle']:
            grid_size = reader.pop('grid_size')
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

        self.save_attention = save_attention
        self.rv2bev_save_attention_path = rv2bev_save_attention_path
        self.bev2rv_save_attention_path = bev2rv_save_attention_path
        self.double_flip = False

        if train_cfg:
            self.bbox_head.set_train_cfg(train_cfg['assigner'])
            self.occupancy = train_cfg['assigner'].get('occupancy', False)
        else:
            self.bbox_head.set_train_cfg(test_cfg['assigner'])
            self.double_flip = test_cfg.get('double_flip', False)
        if not (self.bbox_head.mode in ['bev', 'rv']):
            if 'FHD' in self.backbone.name:
                ds_factor = 16
            else:
                ds_factor = 8
            if grid_size[0] % ds_factor:
                num_input_features = (grid_size[0] // ds_factor + 1) * 128
            else:
                num_input_features = (grid_size[0] // ds_factor) * 128

            # cfg_bev_neck = dict(
            #     type="RPNT",
            #     layer_nums=[5, 5],
            #     ds_layer_strides=[1, 2],
            #     ds_num_filters=[128, 256],
            #     us_layer_strides=[1, 2],
            #     us_num_filters=[256, 128],
            #     num_input_features=1792,  # 0.016:1408, 0.0125:1792 ,0.01: 2304, 0.008: 2944 6144
            #     norm_cfg=None,
            #     logger=logging.getLogger("RPN"),
            # )
            # cfg_rv_neck = dict(
            #     type=cfg_bev_neck['type'],
            #     layer_nums=[i for i in cfg_bev_neck['layer_nums']],
            #     ds_layer_strides=[i for i in cfg_bev_neck['ds_layer_strides']],
            #     ds_num_filters=[i for i in cfg_bev_neck['ds_num_filters']],
            #     us_layer_strides=[i for i in cfg_bev_neck['us_layer_strides']],
            #     us_num_filters=[i for i in cfg_bev_neck['us_num_filters']],
            #     num_input_features=num_input_features,
            #     norm_cfg=cfg_bev_neck['norm_cfg'],
            #     logger=logging.getLogger("RVRPN")
            # )
            # self.bev_unet = builder.build_neck(cfg_bev_neck)
            # self.rv_unet = builder.build_neck(cfg_rv_neck)

            # Cross Attention Module Initialization
            # self.rv2bev_ca_layer = builder.build_attention(atten.rv2bev_atten)
            # self.bev2rv_ca_layer = builder.build_attention(atten.bev2rv_atten)

            if backbone.mode in ['rv2bev', 'cycle']:
                if grid_size[0] % 8:
                    self.rv2bev = nn.Conv2d(
                        grid_size[-1] // 8, grid_size[0] // 8 + 1, kernel_size=1)
                    #self.rv2bev = nn.MaxPool2d(grid_size[-1] // 8)
                    #self.rv2bev = nn.AvgPool2d(grid_size[-1] // 8)
                else:
                    self.rv2bev = nn.Conv2d(grid_size[-1] // 8, grid_size[0] // 8, kernel_size=1)
                #self.fuse_layer = nn.Conv2d(grid_size[-1] // 8, grid_size[0] // 8 + 1, kernel_size=1)
            if backbone.mode in ['bev2rv', 'cycle']:
                if grid_size[0] % 8:
                    self.bev2rv = nn.Conv2d(grid_size[0] // 8 + 1,
                                            grid_size[-1] // 8, kernel_size=1)
                else:
                    self.bev2rv = nn.Conv2d(grid_size[0] // 8, grid_size[-1] // 8, kernel_size=1)
            # self.rv_layers = nn.Sequential(
            # nn.Conv2d(8192, 8192//4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8192//4),
            # nn.ReLU(),
            # nn.Conv2d(8192//4, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            # nn.ReLU(),
            # )

    def extract_feat(self, data, return_loss):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if (self.bbox_head.mode in ['bev', 'rv']) and self.with_neck:
            return self.neck(x)
        else:
            x_bev, x_rv = x
            #print(x_bev.shape, x_rv.shape)
            # x_bev = self.bev_unet(x_bev)
            # x_rv = self.rv_unet(x_rv)
            # N, C, D, H, W = x.shape
            # x_bev = self.neck(x.view(N, C * D, H, W))
            # x = x.permute(0,1,4,3,2)
            # x_rv = self.rv_layers(x.contiguous().view(N, C * W, H, D))
            x_bev, x_rv, x_rv_atten_map, x_bev_atten_map = self.neck((x_bev, x_rv))
            if return_loss or self.backbone.mode in ['rv2bev', 'cycle']:
                # x_rv = self.rv_neck(x_rv)
                # print(x_bev.shape, x_rv.shape)
                # x_bev_atten, x_bev_atten_map = self.rv2bev_ca_layer[0]((x_bev, x_rv))
                # x_bev_atten = self.rv2bev_ca_layer[1](x_bev_atten)
                # x_rv_atten, _ = self.bev2rv_ca_layer[0]((x_rv, x_bev))
                # x_rv_atten = self.bev2rv_ca_layer[1](x_rv_atten)
                return (x_bev, x_rv), x_rv_atten_map, x_bev_atten_map
            return x_bev, x_bev

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
        x, x_rv_atten_map, x_bev_atten_map = self.extract_feat(data, return_loss)

        if not return_loss:
            if self.save_attention:
                mkdir(self.rv2bev_save_attention_path)
                mkdir(self.bev2rv_save_attention_path)
                atten_map_rv = x_rv_atten_map.cpu().detach().numpy()
                atten_map_bev = x_bev_atten_map.cpu().detach().numpy()
                for index, metadata in enumerate(example['metadata']):
                    token = metadata['token']
                    atten_rv = atten_map_rv[index]
                    np.save(os.path.join(self.rv2bev_save_attention_path, '%s.npy' % token), atten_rv)
                    atten_bev = atten_map_bev[index]
                    np.save(os.path.join(self.bev2rv_save_attention_path, '%s.npy' % token), atten_bev)

        grid_size = example['shape'][0]
        preds = self.bbox_head(x, return_loss)
        del data['features']
        del example["voxels"]
        # torch.cuda.empty_cache()

        if self.bbox_head.mode == 'rv2bev':
            for i in range(len(preds)):
                if return_loss:
                    preds[i]['combine_score'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
                else:
                    # for k in range(batch_size):
                    #     shabi = torch.sigmoid(preds[i]['cls_preds'][k]).cpu().numpy()
                    #     np.save(os.path.join('bev_map3', example['metadata'][k]['token']) + '_%i' % i, shabi)
                    preds[i]['cls_preds'] = 0.2 * self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3))\
                        .permute(0, 2, 1, 3) + 0.8 * preds[i]['cls_preds']
                    # if using max/avg pooling
                    # preds[i]['cls_preds'] = 0.2 * self.rv2bev(preds[i]['rv_cls_preds']).repeat(1,1,preds[i]['cls_preds'].shape[2],1) \
                    #      + 0.8 * preds[i]['cls_preds']

                    # preds[i]['cls_preds'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                    #     .permute(0, 2, 1, 3).contiguous()
                    # preds[i]['cls_preds'] = 0.2 * self.fuse_layer(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                    #      .permute(0, 2, 1, 3) + 0.8 * preds[i]['cls_preds']
        if self.bbox_head.mode == 'bev2rv':
            for i in range(len(preds)):
                if return_loss:
                    preds[i]['combine_score'] = self.bev2rv(preds[i]['cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
        if self.bbox_head.mode == 'cycle':
            for i in range(len(preds)):
                if return_loss:
                    preds[i]['combine_score'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
                    preds[i]['bev2rv_score'] = self.bev2rv(preds[i]['cls_preds'].permute(0, 2, 1, 3)) \
                        .permute(0, 2, 1, 3)
                else:
                    # pass
                    preds[i]['cls_preds'] = 0.2 * self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3))\
                        .permute(0, 2, 1, 3) + 0.8 * preds[i]['cls_preds']
                    #preds[i]['cls_preds'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3))
                    # preds[i]['combine_score'] = self.rv2bev(preds[i]['rv_cls_preds'].permute(0, 2, 1, 3)) \
                    #     .permute(0, 2, 1, 3)
                    # preds[i]['bev2rv_score'] = self.bev2rv(preds[i]['cls_preds'].permute(0, 2, 1, 3)) \
                    #     .permute(0, 2, 1, 3)
                # for k in range(batch_size):
                #     shabi = torch.sigmoid(preds[i]['cls_preds'][k]).cpu().numpy()
                #     np.save(os.path.join('bev_map',example['metadata'][k]['token'])+'_%i'%i, shabi)
                #
                #     shabi = torch.sigmoid(preds[i]['combine_score'][k]).cpu().numpy()
                #     np.save(os.path.join('rv_map', example['metadata'][k]['token']) + '_%i'%i, shabi)
                #
                    # shabi = torch.sigmoid(0.2 * preds[i]['combine_score'][k] + 0.8 * preds[i]['cls_preds'][k]).cpu().numpy()
                    # np.save(os.path.join('cycle_map', example['metadata'][k]['token']) + '_%i'%i, shabi)
        if return_loss:
            if self.occupancy:
                occupancy = spconv.SparseConvTensor(
                    torch.ones((len(coordinates), 1), device=coordinates.device,
                               dtype=x.dtype), coordinates.int(),
                    grid_size[::-1], batch_size).dense().squeeze(1)
                occupancy = nn.AdaptiveMaxPool2d(x.shape[-2:])(occupancy).detach()
                occupancy, _ = torch.max(occupancy, dim=1)
                occupancy = occupancy.bool()
            else:
                occupancy = None
            return self.bbox_head.loss(example, preds, occupancy)
        else:
            # self.bbox_head.loss(example, preds, occupancy=None)

            # print(self.bbox_head.ohs_loss[0].shabi_box[0])
            # print(self.bbox_head.ohs_loss[0].shabi_loc[0])
            # # for k in range(batch_size):
            # #     for i in range(len(self.bbox_head.ohs_loss)):
            # #         shabi = ((self.bbox_head.ohs_loss[i].shabi_score.view(preds[i]['cls_preds'].shape))[k]).cpu().numpy()
            # #         np.save(os.path.join('gt_map', example['metadata'][k]['token']) + '_%i'%i, shabi)
            # for i in range(len(self.bbox_head.ohs_loss)):
            #     preds[i]['cls_preds'] = self.bbox_head.ohs_loss[i].shabi_score.view(preds[i]['cls_preds'].shape)
            #     preds[i]['box_preds'][self.bbox_head.ohs_loss[i].shabi_loc[:, 0], self.bbox_head.ohs_loss[i].shabi_loc[:,1], self.bbox_head.ohs_loss[i].shabi_loc[:,2], -2:]=self.bbox_head.ohs_loss[i].shabi_box[:,-2:]
            #     #assert (preds[i]['box_preds'][self.bbox_head.ohs_loss[i].shabi_loc[:, 0], self.bbox_head.ohs_loss[i].shabi_loc[:,1], self.bbox_head.ohs_loss[i].shabi_loc[:,2],3:6]>0).all() , "size should be larger than 0"
            # gt_len = 0
            # for i in example['annos'][0]['gt_boxes']:
            #     gt_len += len(i)
            # assert (len(shabi[0]['box3d_lidar']) <= gt_len)
            return self.bbox_head.predict(example, preds, self.test_cfg)
