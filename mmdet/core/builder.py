from mmcv.utils import Registry, build_from_cfg

PRIOR_GENERATORS = Registry('Generator for anchors and points')

ANCHOR_GENERATORS = PRIOR_GENERATORS

BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')

IOU_CALCULATORS = Registry('IoU calculator')


def build_iou_calculator(cfg, default_args=None):
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)


def build_assigner(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    return build_from_cfg(cfg, BBOX_CODERS, default_args)


def build_prior_generator(cfg, default_args=None):
    return build_from_cfg(cfg, PRIOR_GENERATORS, default_args)


def build_anchor_generator(cfg, default_args=None):
    return build_prior_generator(cfg, default_args=default_args)
