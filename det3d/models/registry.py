from det3d.utils import Registry

READERS = Registry("reader")
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
ROI_EXTRACTORS = Registry("roi_extractor")
SHARED_HEADS = Registry("shared_head")
HEADS = Registry("head")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
