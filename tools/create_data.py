import copy
from pathlib import Path
import pickle
from tqdm import tqdm
import fire
import numpy as np
import itertools
from shutil import copyfile
import os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database



def resample_infos(train_info_path, val_info_path, nsweeps):
    tasks = [
        dict(num_class=1, class_names=["car"]),
        dict(num_class=2, class_names=["truck", "construction_vehicle"]),
        dict(num_class=2, class_names=["bus", "trailer"]),
        dict(num_class=1, class_names=["barrier"]),
        dict(num_class=2, class_names=["motorcycle", "bicycle"]),
        dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
    ]

    _class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
    # store n sweeps for faster read:
    with open(val_info_path, "rb") as f:
        _val_infos_all = pickle.load(f)
    for info in tqdm(_val_infos_all):
        nu_ds.combine_sweeps(info, nsweeps)
    with open(val_info_path, "wb") as f:
        pickle.dump(_val_infos_all, f)

    with open(train_info_path, "rb") as f:
        _train_infos_all = pickle.load(f)
    for info in tqdm(_train_infos_all):
        nu_ds.combine_sweeps(info, nsweeps)

    frac = int(len(_train_infos_all) * 0.25)

    _cls_infos = {name: [] for name in _class_names}
    for info in _train_infos_all:
        for name in set(info["gt_names"]):
            if name in _class_names:
                _cls_infos[name].append(info)

    duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
    _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

    new_train_infos = []

    frac = 1.0 / len(_class_names)
    ratios = [frac / v for v in _cls_dist.values()]

    for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
        new_train_infos += np.random.choice(
            cls_infos, int(len(cls_infos) * ratio)
        ).tolist()

    _cls_infos = {name: [] for name in _class_names}
    for info in new_train_infos:
        for name in set(info["gt_names"]):
            if name in _class_names:
                _cls_infos[name].append(info)

    new_train_info_path = train_info_path.replace('velo.pkl', 'velo_resampled.pkl')
    with open(new_train_info_path, "wb") as f:
        pickle.dump(new_train_infos, f)


def nuscenes_data_prep(root_path, version, nsweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps)
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(root_path) / "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps),
        nsweeps=1,
    )
    train_info_read_path = Path(root_path) / "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps)
    train_info_copy_path = Path(root_path) / \
        "infos_train_{:02d}sweeps_repeat_withvelo.pkl".format(nsweeps)
    val_info_read_path = Path(root_path) / "infos_val_{:02d}sweeps_withvelo.pkl".format(nsweeps)
    val_info_copy_path = Path(root_path) / \
        "infos_val_{:02d}sweeps_repeat_withvelo.pkl".format(nsweeps)
    copyfile(train_info_read_path, train_info_copy_path)
    copyfile(val_info_read_path, val_info_copy_path)
    os.mkdir(os.path.join(root_path, 'samples_10LIDAR_TOP'))
    resample_infos(train_info_copy_path, val_info_copy_path, nsweeps)


def nuscenes_data_prep_test(root_path, nsweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version='v1.0-test', nsweeps=nsweeps)
    val_info_read_path = Path(root_path) / "infos_test_10sweeps_withvelo.pkl"
    val_info_dump_path = Path(root_path) / "infos_test_10sweeps_repeat_withvelo.pkl"
    with open(val_info_read_path, "rb") as f:
        _val_infos_all = pickle.load(f)
    for info in tqdm(_val_infos_all):
        nu_ds.combine_sweeps(info, nsweeps)
    with open(val_info_dump_path, "wb") as f:
        pickle.dump(_val_infos_all, f)
    # nu_ds.get_sample_ground_plane(root_path, version=version)


if __name__ == "__main__":
    fire.Fire()
