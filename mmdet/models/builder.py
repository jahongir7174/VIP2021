import collections
import math

import mmcv
import torch
from mmcv.cnn.bricks.activation import ACTIVATION_LAYERS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.utils import Registry

MODELS = Registry('models', parent=mmcv.cnn.MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    assert cfg.get('train_cfg') is None or train_cfg is None, 'train_cfg specified in both outer field and model field'
    assert cfg.get('test_cfg') is None or test_cfg is None, 'test_cfg specified in both outer field and model field'
    return DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


@ACTIVATION_LAYERS.register_module()
class SiLU(torch.nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.silu = torch.nn.SiLU(inplace)

    def forward(self, x):
        return self.silu(x)


@DETECTORS.register_module()
class Detector(BaseModule):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = False
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)})!= num of image meta ({len(img_metas)})')

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, f'aug test does not support inference with batch size {imgs[0].size(0)}'
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    @staticmethod
    def _parse_losses(losses):
        log_vars = collections.OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                loss_value = loss_value.data.clone()
                torch.distributed.all_reduce(loss_value.div_(torch.distributed.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      loss_weights=None,
                      **kwargs):
        xs = self.extract_feat(img)

        if not isinstance(xs[0], (list, tuple)):
            xs = [xs]
            loss_weights = None
        elif loss_weights is None:
            loss_weights = [0.5] + [1] * (len(xs) - 1)  # Reference CBNet paper

        def upd_loss(_losses, idx, weight):
            new_losses = dict()
            for k, v in _losses.items():
                new_k = '{}{}'.format(k, idx)
                if weight != 1 and 'loss' in k:
                    new_k = '{}_w{}'.format(new_k, weight)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            for i, x in enumerate(xs):
                rpn_losses, proposal_list = self.rpn_head.forward_train(x,
                                                                        img_metas,
                                                                        gt_bboxes,
                                                                        gt_labels=None,
                                                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                                                        proposal_cfg=proposal_cfg)
                if len(xs) > 1:
                    rpn_losses = upd_loss(rpn_losses, idx=i, weight=loss_weights[i])
                losses.update(rpn_losses)
        else:
            proposal_list = proposals

        for i, x in enumerate(xs):
            roi_losses = self.roi_head.forward_train(x,
                                                     img_metas,
                                                     proposal_list,
                                                     gt_bboxes,
                                                     gt_labels,
                                                     gt_bboxes_ignore,
                                                     gt_masks,
                                                     **kwargs)
            if len(xs) > 1:
                roi_losses = upd_loss(roi_losses, idx=i, weight=loss_weights[i])
            losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale)
