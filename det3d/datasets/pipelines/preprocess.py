import numpy as np

from det3d import torchie
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import (
    build_dbsampler,
)
from det3d.core.input.voxel_generator import VoxelGenerator

from ..registry import PIPELINES


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.mode = cfg.mode
        self.random_crop = cfg.get("random_crop", False)
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_noise_std = cfg.global_trans_noise
            self.gt_points_drop = cfg.gt_drop_percentage
            self.gt_drop_max_keep = cfg.gt_drop_max_keep_points
            self.remove_points_after_sample = cfg.remove_points_after_sample
            self.class_names = cfg.class_names
            if cfg.db_sampler and cfg.db_sampler.enable:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None
            self.npoints = cfg.get("npoints", -1)
            self.flip_single = cfg.get("flip_single", False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] == "NuScenesDataset":
            points = res["lidar"]["combined"]

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]
            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }
            if res["type"] == "WaymoDataset":
                gt_dict['num_points_in_gt'] = np.array(anno_dict["num_points_in_gt"]).reshape(-1)
            if "difficulty" not in anno_dict:
                difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
                gt_dict["difficulty"] = difficulty
            else:
                gt_dict["difficulty"] = anno_dict["difficulty"]

        if "calib" in res:
            calib = res["calib"]
        else:
            calib = None

        if self.mode == "train":
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore"]
            )

            _dict_select(gt_dict, selected)
            gt_dict.pop("difficulty")

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    # res["metadata"]["num_point_features"],
                    points.shape[-1],
                    self.random_crop,
                    gt_group_ids=None,
                    calib=calib,
                    gt_drop_rate=self.gt_points_drop,
                    gt_drop_max_keep=self.gt_drop_max_keep,
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    if self.remove_points_after_sample:
                        masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                        points = points[np.logical_not(masks.any(-1))]

                    points = np.concatenate([sampled_points, points], axis=0)
                    if self.gt_drop_max_keep == -1:
                        inds = np.random.choice(range(len(points)), int(
                            len(points) * 0.1), replace=False)
                        rand_nums = np.random.uniform(0, 31, int(len(points) * 0.1))
                        points[inds, -1] = rand_nums

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes
            if len(gt_dict["gt_boxes"]):
                if self.flip_single:
                    gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
                else:
                    gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
                gt_dict["gt_boxes"], points = prep.global_rotation(
                    gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
                )
                gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                    gt_dict["gt_boxes"], points, *self.global_scaling_noise
                )

        if self.shuffle_points:
            # shuffle is a little slow.
            np.random.shuffle(points)

        res["lidar"]["points"] = points
        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = cfg.max_voxel_num
        self.drop_rate = cfg.get('drop_rate', 0)
        self.faster = cfg.get('faster', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
            faster=self.faster
        )
        self.double_flip = cfg.get('double_flip', False)

    def __call__(self, res, info):
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        # [352, 400]
        double_flip = self.double_flip and (res["mode"] != 'train')
        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            h_range = pc_range[[2, 5]]
            if len(gt_dict["gt_boxes"]):
                mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
                _dict_select(gt_dict, mask)
            res["lidar"]["annotations"] = gt_dict

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"]
        )
        if (res["mode"] == "train") and (self.drop_rate > 0):
            voxel_mask = np.random.uniform(size=len(voxels))
            voxel_mask = voxel_mask >= self.drop_rate
            voxels = voxels[voxel_mask]
            coordinates = coordinates[voxel_mask]
            num_points = num_points[voxel_mask]
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
        )

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

        return res, info


@PIPELINES.register_module
class AssignHotspots(object):
    def __init__(self, **kwargs):
        assigner_cfg = kwargs["cfg"]
        target_assigner_config = assigner_cfg.target_assigner
        self.tasks = target_assigner_config.tasks

    def __call__(self, res, info):

        class_names_by_task = [t.class_names for t in self.tasks]

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)
            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                if len(task_box):
                    task_box[:, -1] = box_np_ops.limit_period(
                        task_box[:, -1], offset=0.5, period=np.pi * 2
                    )
            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

        return res, info
