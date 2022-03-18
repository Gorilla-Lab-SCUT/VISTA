# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Sequence
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.bbox.box_coders import GroundBox3dCoderAF
from ...core.bbox import box_torch_ops
from ...core.bbox.geometry import points_in_convex_polygon_torch


class AttentionConstrainedLoss(nn.Module):
    def __init__(self,
                 box_coder: GroundBox3dCoderAF,
                 num_class: int,
                 task_id: Sequence[int],
                 query_res: Sequence[int],
                 loss_weight: int = 1.0):
        super().__init__()
        self.cls_out_channels = num_class
        self.pc_range = np.asarray(box_coder.pc_range)
        self.dims = self.pc_range[3:] - self.pc_range[:3]
        self.task_id = task_id
        self.query_res = query_res
        self._loss_weight = loss_weight

        self.h, self.w = self.query_res[0], self.query_res[1]
        ww, hh = np.meshgrid(range(self.w), range(self.h))
        ww = ww.reshape(-1)
        hh = hh.reshape(-1)
        self.ww_l = torch.LongTensor(ww).to(torch.cuda.current_device())
        self.hh_l = torch.LongTensor(hh).to(torch.cuda.current_device())
        ww = torch.FloatTensor(ww).to(torch.cuda.current_device()) + 0.5
        hh = torch.FloatTensor(hh).to(torch.cuda.current_device()) + 0.5
        ww = ww / self.w * self.dims[0] + self.pc_range[0]
        hh = hh / self.h * self.dims[1] + self.pc_range[1]
        self.grids_sensor = torch.stack([ww, hh], 1).clone().detach()
        self.effective_ratio = [1.0, 6.0]

    def find_grid_in_bbx_single(self,
                                x):
        """
        find the attention grids that are enclosed by a GT bounding box
        Args:
            query_res (Sequence[int]): the size of the query feat map
            gt_bboxes (torch.Tensor, [M, ndim]): a single GT bounding boxes set for a scene
        """
        query_res, gt_bboxes = x
        gt_bboxes = torch.from_numpy(gt_bboxes).cuda()
        bboxes_grid_ind_list = []
        if len(gt_bboxes > 0):
            temp_grid_flag = -1 * torch.ones(query_res, dtype=torch.long).cuda()
            effective_boxes = gt_bboxes[:, [0, 1, 3, 4]].clone().detach()  # [M, 4]
            effective_ratio_l = (self.dims[0] / self.w) / effective_boxes[:, 2]  # [M]
            effective_ratio_w = (self.dims[1] / self.h) / effective_boxes[:, 3]  # [M]
            effective_ratio_l = effective_ratio_l.clamp(min=self.effective_ratio[0],  # [M]
                                                        max=self.effective_ratio[1])  # [M]
            effective_ratio_w = effective_ratio_w.clamp(min=self.effective_ratio[0],  # [M]
                                                        max=self.effective_ratio[1])  # [M]
            effective_boxes[:, 2] *= effective_ratio_l
            effective_boxes[:, 3] *= effective_ratio_w
            angles = gt_bboxes[:, -1]
            effective_boxes = box_torch_ops.center_to_corner_box2d(
                effective_boxes[:, :2], effective_boxes[:, 2:4], angles)
            grid_real_centers = self.grids_sensor
            w_indices = self.ww_l
            h_indices = self.hh_l
            for i in range(len(gt_bboxes)):
                pos_mask = points_in_convex_polygon_torch(
                    grid_real_centers, effective_boxes[i].unsqueeze(0))  # [num_points, 8]
                pos_ind = pos_mask.nonzero()[:, 0]
                gt_center = gt_bboxes[i: i + 1, :2]  # [1, 2]
                dist_to_grid_center = torch.norm(grid_real_centers - gt_center, dim=1)  # [W * H]
                min_ind = torch.argmin(dist_to_grid_center)
                if min_ind not in pos_ind:
                    pos_ind = torch.cat([pos_ind.reshape(-1, 1), min_ind.reshape(-1, 1)],
                                        dim=0).reshape(-1)
                pos_h_indices = h_indices[pos_ind]  # [num_pos]
                pos_w_indices = w_indices[pos_ind]  # [num_pos]
                if len(pos_h_indices):
                    if not (temp_grid_flag[pos_h_indices, pos_w_indices] == -1).all():
                        unique_pos_h_indices = pos_h_indices.new_zeros((0,))
                        unique_pos_w_indices = pos_w_indices.new_zeros((0,))
                        for ph, pw in zip(pos_h_indices, pos_w_indices):
                            if temp_grid_flag[ph, pw] == -1:
                                unique_pos_h_indices = torch.cat(
                                    (unique_pos_h_indices, ph.view((1))))
                                unique_pos_w_indices = torch.cat(
                                    (unique_pos_w_indices, pw.view((1))))
                            else:
                                temp_grid_flag[ph, pw] = -1
                        pos_h_indices = unique_pos_h_indices
                        pos_w_indices = unique_pos_w_indices
                    temp_grid_flag[pos_h_indices, pos_w_indices] = i
            temp_grid_flag = temp_grid_flag.view(-1)
            for i in range(len(gt_bboxes)):
                bbx_grid_ind = torch.where(temp_grid_flag == i)[0]
                bboxes_grid_ind_list.append(bbx_grid_ind)
        return bboxes_grid_ind_list

    def find_grid_in_bbx(self,
                         gt_bboxes: List[np.ndarray]):
        query_sizes = [self.query_res for i in range(len(gt_bboxes))]
        map_results = map(self.find_grid_in_bbx_single, zip(query_sizes, gt_bboxes))
        return map_results

    def compute_var_loss(self,
                         atten_map: torch.Tensor,
                         grid_ind_list: List[torch.Tensor]):
        var_loss = 0.0
        var_pos_num = 0.0
        for i in range(len(grid_ind_list)):
            grid_ind = grid_ind_list[i]
            temp_var_loss = 0.0
            if len(grid_ind) > 0:
                atten_score = atten_map[grid_ind, :]
                var = torch.var(atten_score, 1).mean()
                temp_var_loss = temp_var_loss + (0.0 - var)
                var_pos_num += 1
            var_loss = var_loss + temp_var_loss
        return var_loss, var_pos_num

    def forward(self,
                atten_map: torch.Tensor,
                gt_bboxes: List[np.ndarray],
                gt_labels: List[np.ndarray],
                **kwargs):
        ret_dict = {}
        batch_grid_ind_list = list(self.find_grid_in_bbx(gt_bboxes))
        var_loss = torch.tensor(0.0).cuda()
        var_pos_num = 0.0
        for i in range(len(gt_bboxes)):
            grid_ind_list = batch_grid_ind_list[i]
            if len(grid_ind_list) > 0:
                temp_var_loss, temp_var_pos_num = self.compute_var_loss(
                    atten_map[i], grid_ind_list)
                var_loss = var_loss + temp_var_loss
                var_pos_num += temp_var_pos_num
        var_pos_num = max(var_pos_num, 1)
        norm_var_loss = var_loss * 1.0 / var_pos_num
        ret_dict["var_loss"] = norm_var_loss
        return ret_dict
