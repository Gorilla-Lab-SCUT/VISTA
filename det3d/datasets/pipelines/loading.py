import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops

from ..registry import PIPELINES


def read_file(path, tries=2, num_point_feature=4):
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            points = np.fromfile(path, dtype=np.float32)
            s = points.shape[0]
            if s % 5 != 0:
                points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep):
    min_distance = 1.0

    points_sweep = read_file(str(sweep["lidar_path"])).T

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    # points_sweep[3, :] /= 255
    points_sweep = remove_close(points_sweep, min_distance)
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "KittiDataset":

            pc_info = info["point_cloud"]
            velo_path = Path(pc_info["velodyne_path"])
            if not velo_path.is_absolute():
                velo_path = (
                    Path(res["metadata"]["image_prefix"]) / pc_info["velodyne_path"]
                )
            velo_reduced_path = (
                velo_path.parent.parent
                / (velo_path.parent.stem + "_reduced")
                / velo_path.name
            )
            if velo_reduced_path.exists():
                velo_path = velo_reduced_path
            points = np.fromfile(str(velo_path), dtype=np.float32, count=-1).reshape(
                [-1, res["metadata"]["num_point_features"]]
            )

            res["lidar"]["points"] = points
        elif self.type == "NuScenesDataset":

            lidar_path = info["lidar_path"]
            
            points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])

            res["lidar"]["combined"] = points

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:

            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        else:
            if "gt_boxes" in info:
                return NotImplementedError
        return res, info
