import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

norm_cfg = None
DOUBLE_FLIP = False

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
voxel_generator = dict(
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    voxel_size=[0.08, 0.08, 0.08],
    max_points_in_voxel=6,
    max_voxel_num=5000000,
    double_flip=DOUBLE_FLIP,
    faster=True
)

# training and testing settings
target_assigner = dict(
    type="iou",
    tasks=tasks,
)
box_coder_args = dict(
    x_range=[-4.5, 4.5], y_range=[-4.5, 4.5], z_range=[-5, 3], xy_bin_num=16, z_bin_num=12, r_bin_num=12, dim_bin_num=12, dim_range=[-3, 3],
)

box_coder = dict(
    type="ground_box3d_coder_anchor_free", velocity=True, center='soft_argmin', height='bottom_soft_argmin', dim='log_soft_argmin', rotation='vector', pc_range=voxel_generator["range"], kwargs=box_coder_args,
)

# model settings
model = dict(
    type="OHS_Multiview",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(
        type="MultiViewBackbone", num_input_features=5, ds_factor=8, norm_cfg=norm_cfg,
    ),
    neck=dict(
        type="RPNT",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 128],
        num_input_features=640,  # 0.08:640, 0.1:512, 0.125:384, 0.2:256
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="DeepMultiGroupOHSHeadClear_Decouple",
        mode="bev",
        in_channels=sum([128, 256]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1, ],
        atten_res=(40, 40),
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=2.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25,
                      gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedL1Loss",
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            codewise=True,
            loss_weight=0.25,
        ),
    ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
    effective_ratio=[1.0, 8.0],
    ignore_ratio_in=0,
    ignore_ratio_out=8.0,
    select_hotspots=True,
    num_hotspots=28,
    occupancy=False,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=80,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    assigner=assigner,
    double_flip=DOUBLE_FLIP
)

# dataset settings
dataset_type = "NuScenesDataset"
n_sweeps = 10
data_root = "/data/dataset/nuscenes"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path="/data/dataset/nuscenes/dbinfos_train_1sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        dict(truck=3),
        dict(construction_vehicle=7),
        dict(bus=4),
        dict(trailer=6),
        dict(barrier=6),
        dict(motorcycle=2),
        dict(bicycle=6),
        dict(pedestrian=2),
        dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
    gt_drop_percentage=0.5,
    gt_drop_max_keep_points=5,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)


train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignHotspots", cfg=train_cfg["assigner"]),
    dict(type="ReformatOHS"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
if DOUBLE_FLIP:
    test_pipeline = [
        dict(type="LoadPointCloudFromFile", dataset=dataset_type),
        dict(type="LoadPointCloudAnnotations", with_bbox=True),
        dict(type="Preprocess", cfg=val_preprocessor),
        dict(type="DoubleFlip"),
        dict(type="Voxelization", cfg=voxel_generator),
        dict(type="AssignHotspots", cfg=train_cfg["assigner"]),
        dict(type="ReformatOHS", double_flip=DOUBLE_FLIP),
    ]
else:
    test_pipeline = [
        dict(type="LoadPointCloudFromFile", dataset=dataset_type),
        dict(type="LoadPointCloudAnnotations", with_bbox=True),
        dict(type="Preprocess", cfg=val_preprocessor),
        dict(type="Voxelization", cfg=voxel_generator),
        dict(type="AssignHotspots", cfg=train_cfg["assigner"]),
        dict(type="ReformatOHS"),
    ]

#train_anno = "data/Nuscenes/v1.0-trainval/infos_train_10sweeps_withvelo_resampled.pkl"
train_anno = "/data/dataset/nuscenes/infos_train_10sweeps_repeat_withvelo_resampled.pkl"
val_anno = "/data/dataset/nuscenes/infos_val_10sweeps_repeat_withvelo.pkl"
test_anno = None

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.0010, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = ""
load_from = None
resume_from = ''
workflow = [("train", 1), ]
# workflow = [('train', 1)]
