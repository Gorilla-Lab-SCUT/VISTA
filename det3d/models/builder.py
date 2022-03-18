from det3d.utils import build_from_cfg
from det3d.models.utils.norm import build_norm_layer
from torch import nn

from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
    ROI_EXTRACTORS,
    SHARED_HEADS,
)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_reader(cfg):
    return build(cfg, READERS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


class None_Class():
    def __init__(self) -> None:
        pass


def build_attention(cfg):
    norm_cfg = cfg.pop("norm_cfg")
    act = cfg.pop("act", None_Class)
    atten_module = build_neck(cfg)
    atten_norm = build_norm_layer(norm_cfg, None)[1]
    if issubclass(act, nn.Module):
        act_layer = act()
        return nn.Sequential(atten_module, atten_norm, act_layer)
    else:
        return nn.Sequential(atten_module, atten_norm)
