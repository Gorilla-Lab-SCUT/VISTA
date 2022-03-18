from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import torch
from . import box_np_ops, box_torch_ops
import pdb
import copy
class BoxCoder(object):
    """Abstract base class for box coder."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass

class GroundBox3dCoderAF(BoxCoder):
    def __init__(self, velocity=True, center='direct', height='direct', dim='log', rotation='direct', pc_range=[-50, -51.2, -5, 50, 51,2, 3],kwargs=None):
        super().__init__()
        self.velocity = velocity
        self.center = center
        self.height = height
        self.dim = dim
        self.rotation = rotation
        self.pc_range = np.array(pc_range)
        self.dims = self.pc_range[3:] - self.pc_range[:3]
        self.n_dim = 7
        self.kwargs = kwargs
        if velocity: self.n_dim += 2
        if rotation == 'vector': self.n_dim += 1
        self.grids_sensor = None
        self.ww_l = None
        self.hh_l = None

    @property
    def code_size(self):
        return self.n_dim

    def layout(self, w, h):
        if self.grids_sensor is None:
            mode = self.kwargs.get('mode', None)

            ww, hh = np.meshgrid(range(w), range(h))
            ww = ww.reshape(-1)
            hh = hh.reshape(-1)
            self.ww_l = torch.LongTensor(ww).to(torch.cuda.current_device())
            self.hh_l = torch.LongTensor(hh).to(torch.cuda.current_device())

            ww = torch.FloatTensor(ww).to(torch.cuda.current_device()) + 0.5
            hh = torch.FloatTensor(hh).to(torch.cuda.current_device()) + 0.5
            ww = ww / w * self.dims[0] + self.pc_range[0]
            hh = hh / h * self.dims[1] + self.pc_range[1]

            self.grids_sensor = torch.stack([ww, hh], 1).clone().detach()

    def _encode(self, centers, shifts, gt_bbox):
        shifts[:, 0:2] = gt_bbox[0:2] - centers
        if 'bottom' in self.height:
            shifts[:, 2] = gt_bbox[2] - 0.5 * gt_bbox[5]
        else:
            shifts[:, 2] = gt_bbox[2]
        if self.rotation == 'vector':
            shifts[:, -2] = torch.cos(gt_bbox[-1])
            shifts[:, -1] = torch.sin(gt_bbox[-1])
        else:
            shifts[:, -1] = gt_bbox[-1]

        if 'log' in self.dim:
            shifts[:, 3:6] = torch.log(gt_bbox[3:6])
        else:
            shifts[:, 3:6] = gt_bbox[3:6]
        if self.velocity:
            shifts[:, 6:8] = gt_bbox[6:8]
        return shifts

    def _decode(self, shifts, w, h):
        self.layout(w, h)
        shifts[:, :, 0] += self.grids_sensor[:, 0]
        shifts[:, :, 1] += self.grids_sensor[:, 1]
        if self.rotation == 'vector':
            shifts[:, :, -2] = torch.atan2(shifts[:, :, -1], shifts[:, :, -2])
            shifts = shifts[:, :, :-1]
        if 'log' in self.dim:
            shifts[:, :, 3:6] = torch.exp(shifts[:, :, 3:6])
        if 'bottom' in self.height:
            shifts[:, :, 2] += 0.5 * shifts[:, :, 5]
        return shifts


