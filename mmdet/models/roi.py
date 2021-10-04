import logging
import warnings

import numpy
import torch
from mmcv import ops
from mmcv.cnn import build_conv_layer, build_upsample_layer, ConvModule
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.utils import Registry

from mmdet.core.builder import (build_assigner, build_bbox_coder, build_sampler)
from mmdet.core.mask import mask_target
from mmdet.core.util import bbox2result
from mmdet.models.builder import ROI_EXTRACTORS, build_loss
from mmdet.models.util import accuracy, bbox_flip, multi_apply, multiclass_nms, bbox_mapping_back
from .builder import HEADS, build_head, build_roi_extractor, build_shared_head

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

logger = logging.getLogger(__name__)

TRANSFORMER = Registry('Transformer')
LINEAR_LAYERS = Registry('linear layers')

LINEAR_LAYERS.register_module('Linear', module=torch.nn.Linear)


def box2roi(bbox_list):
    rois_list = []
    for img_id, boxes in enumerate(bbox_list):
        if boxes.size(0) > 0:
            img_indices = boxes.new_full((boxes.size(0), 1), img_id)
            rois = torch.cat([img_indices, boxes[:, :4]], dim=-1)
        else:
            rois = boxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def box_mapping(boxes, img_shape, scale_factor, flip, flip_direction='horizontal'):
    new_boxes = boxes * boxes.new_tensor(scale_factor)
    if flip:
        new_boxes = bbox_flip(new_boxes, img_shape, flip_direction)
    return new_boxes


def merge_aug_boxes(aug_boxes, aug_scores, img_metas):
    recovered_boxes = []
    for boxes, img_info in zip(aug_boxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        boxes = bbox_mapping_back(boxes, img_shape, scale_factor, flip, flip_direction)
        recovered_boxes.append(boxes)
    boxes = torch.stack(recovered_boxes).mean(dim=0)
    if aug_scores is None:
        return boxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return boxes, scores


def merge_aug_scores(aug_scores):
    if isinstance(aug_scores[0], torch.Tensor):
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        return numpy.mean(aug_scores, axis=0)


def merge_aug_masks(aug_masks, img_metas, weights=None):
    recovered_masks = []
    for mask, img_info in zip(aug_masks, img_metas):
        flip = img_info[0]['flip']
        flip_direction = img_info[0]['flip_direction']
        if flip:
            if flip_direction == 'horizontal':
                mask = mask[:, :, :, ::-1]
            elif flip_direction == 'vertical':
                mask = mask[:, :, ::-1, :]
            elif flip_direction == 'diagonal':
                mask = mask[:, :, :, ::-1]
                mask = mask[:, :, ::-1, :]
            else:
                raise ValueError(
                    f"Invalid flipping direction '{flip_direction}'")
        recovered_masks.append(mask)

    if weights is None:
        merged_masks = numpy.mean(recovered_masks, axis=0)
    else:
        merged_masks = numpy.average(
            numpy.array(recovered_masks), axis=0, weights=numpy.array(weights))
    return merged_masks


def build_linear_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='Linear')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in LINEAR_LAYERS:
        raise KeyError(f'Unrecognized linear type {layer_type}')
    else:
        linear_layer = LINEAR_LAYERS.get(layer_type)

    return linear_layer(*args, **kwargs, **cfg_)


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device).to(torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device).to(torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    # IsInf op is not supported with ONNX<=1.7.0
    if not torch.onnx.is_in_onnx_export():
        if torch.isinf(img_x).any():
            inds = torch.where(torch.isinf(img_x))
            img_x[inds] = 0
        if torch.isinf(img_y).any():
            inds = torch.where(torch.isinf(img_y))
            img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = torch.nn.functional.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


@HEADS.register_module()
class CascadeRoIHead(BaseModule):
    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(self.num_stages)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        self.mask_head = torch.nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [mask_roi_extractor for _ in range(self.num_stages)]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        # bbox head
        outs = ()
        rois = box2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'],)
        return outs

    def _bbox_forward(self, stage, x, rois):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        rois = box2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(sampling_results,
                                                         gt_bboxes,
                                                         gt_labels,
                                                         rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'],
                                               rois, *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self, stage, x, sampling_results, gt_masks, rcnn_train_cfg, bbox_feats=None):
        pos_rois = box2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'], mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self, x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(proposal_list[j],
                                                         gt_bboxes[j],
                                                         gt_bboxes_ignore[j],
                                                         gt_labels[j])
                    sampling_result = bbox_sampler.sample(assign_result,
                                                          proposal_list[j],
                                                          gt_bboxes[j],
                                                          gt_labels[j],
                                                          feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels, rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(i, x, sampling_results,
                                                        gt_masks, rcnn_train_cfg, bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(cls_score)
                    roi_labels = torch.where(roi_labels == self.bbox_head[i].num_classes,
                                             cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'],
                                                                    roi_labels,
                                                                    bbox_results['bbox_pred'],
                                                                    pos_is_gts, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = box2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [self.bbox_head[i].loss_cls.get_activation(s)
                                 for s in cls_score]
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([self.bbox_head[i].regress_by_class(rois[j],
                                                                     bbox_label[j],
                                                                     bbox_pred[j],
                                                                     img_metas[j])
                                  for j in range(num_imgs)])

        # average scores of each image by stages
        cls_score = [sum([score[i] for score in ms_scores]) / float(len(ms_scores))
                     for i in range(num_imgs)]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(rois[i],
                                                                cls_score[i],
                                                                bbox_pred[i],
                                                                img_shapes[i],
                                                                scale_factors[i],
                                                                rescale=rescale,
                                                                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head[-1].num_classes)
                        for i in range(num_imgs)]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                                     for scale_factor in scale_factors]
                _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i][:, :4]
                           for i in range(len(det_bboxes))]
                mask_rois = box2roi(_bboxes)
                num_mask_rois_per_img = tuple(_bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append([[] for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(aug_mask, [[img_metas[i]]] * self.num_stages)
                        segm_result = self.mask_head[-1].get_seg_masks(merged_masks,
                                                                       _bboxes[i],
                                                                       det_labels[i],
                                                                       rcnn_test_cfg,
                                                                       ori_shapes[i],
                                                                       scale_factors[i],
                                                                       rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = box_mapping(proposal_list[0][:, :4],
                                    img_shape, scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = box2roi([proposals])
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(rois,
                                                           cls_score,
                                                           bbox_results['bbox_pred'],
                                                           img_shape,
                                                           scale_factor,
                                                           rescale=False,
                                                           cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_boxes(aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = box_mapping(det_bboxes[:, :4], img_shape,
                                          scale_factor, flip, flip_direction)
                    mask_rois = box2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas)

                ori_shape = img_metas[0][0]['ori_shape']
                dummy_scale_factor = numpy.ones(4)
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks,
                                                               det_bboxes,
                                                               det_labels,
                                                               rcnn_test_cfg,
                                                               ori_shape,
                                                               scale_factor=dummy_scale_factor,
                                                               rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]


@HEADS.register_module()
class HybridTaskCascadeRoIHead(CascadeRoIHead):
    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 interleaved=True,
                 mask_info_flow=True,
                 **kwargs):
        super().__init__(num_stages, stage_loss_weights, **kwargs)
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow

    def forward_dummy(self, x, proposals):
        outs = ()
        # bbox heads
        rois = box2roi([proposals])
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred,)
        return outs

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg):
        bbox_head = self.bbox_head[stage]
        rois = box2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)

        bbox_targets = bbox_head.get_targets(sampling_results,
                                             gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'],
                                   rois, *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox,
                            rois=rois,
                            bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = box2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs], pos_rois)

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats, return_feat=False)

        mask_targets = mask_head.get_targets(sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)

        mask_results = dict(loss_mask=loss_mask)
        return mask_results

    def _bbox_forward(self, stage, x, rois):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _mask_forward_test(self, stage, x, bboxes):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = box2roi([bboxes])
        mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        losses = dict()

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(assign_result,
                                                      proposal_list[j],
                                                      gt_bboxes[j],
                                                      gt_labels[j],
                                                      feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels, rcnn_train_cfg)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'],
                                                                        roi_labels,
                                                                        bbox_results['bbox_pred'],
                                                                        pos_is_gts,
                                                                        img_metas)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(proposal_list[j],
                                                                 gt_bboxes[j],
                                                                 gt_bboxes_ignore[j],
                                                                 gt_labels[j])
                            sampling_result = bbox_sampler.sample(assign_result,
                                                                  proposal_list[j],
                                                                  gt_bboxes[j],
                                                                  gt_labels[j],
                                                                  feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                mask_results = self._mask_forward_train(i, x, sampling_results,
                                                        gt_masks, rcnn_train_cfg)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(bbox_results['rois'],
                                                                    roi_labels,
                                                                    bbox_results['bbox_pred'],
                                                                    pos_is_gts, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = box2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(i, x, rois)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([bbox_head.regress_by_class(rois[i],
                                                             bbox_label[i],
                                                             bbox_pred[i],
                                                             img_metas[i])
                                  for i in range(num_imgs)])

        # average scores of each image by stages
        cls_score = [sum([score[i] for score in ms_scores]) / float(len(ms_scores))
                     for i in range(num_imgs)]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(rois[i],
                                                                cls_score[i],
                                                                bbox_pred[i],
                                                                img_shapes[i],
                                                                scale_factors[i],
                                                                rescale=rescale,
                                                                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        bbox_result = [bbox2result(det_bboxes[i], det_labels[i], self.bbox_head[-1].num_classes)
                       for i in range(num_imgs)]
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)] for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                                     for scale_factor in scale_factors]
                _bboxes = [det_bboxes[i][:, :4] * scale_factors[i] if rescale else det_bboxes[i]
                           for i in range(num_imgs)]
                mask_rois = box2roi(_bboxes)
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                last_feat = None

                num_bbox_per_img = tuple(len(_bbox) for _bbox in _bboxes)
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)

                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_bbox_per_img, 0)
                    aug_masks.append([mask.sigmoid().cpu().numpy() for mask in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append([[] for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_mask = merge_aug_masks(aug_mask, [[img_metas[i]]] * self.num_stages)
                        segm_result = self.mask_head[-1].get_seg_masks(merged_mask, _bboxes[i],
                                                                       det_labels[i], rcnn_test_cfg,
                                                                       ori_shapes[i], scale_factors[i], rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(img_feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = box_mapping(proposal_list[0][:, :4], img_shape,
                                    scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = box2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label, bbox_results['bbox_pred'], img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(rois,
                                                           cls_score,
                                                           bbox_results['bbox_pred'],
                                                           img_shape,
                                                           scale_factor,
                                                           rescale=False, cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_boxes(aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(img_feats, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = box_mapping(det_bboxes[:, :4], img_shape,
                                          scale_factor, flip, flip_direction)
                    mask_rois = box2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                                                             mask_rois)
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(merged_masks,
                                                               det_bboxes,
                                                               det_labels,
                                                               rcnn_test_cfg,
                                                               ori_shape,
                                                               scale_factor=1.0,
                                                               rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]


@HEADS.register_module()
class BoxHead(BaseModule):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=None,
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=None,
                 cls_predictor_cfg=None,
                 loss_cls=None,
                 loss_bbox=None):
        super().__init__()
        if loss_bbox is None:
            loss_bbox = dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        if loss_cls is None:
            loss_cls = dict(type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0)
        if cls_predictor_cfg is None:
            cls_predictor_cfg = dict(type='Linear')
        if reg_predictor_cfg is None:
            reg_predictor_cfg = dict(type='Linear')
        if bbox_coder is None:
            bbox_coder = dict(type='DeltaXYWHBBoxCoder',
                              clip_border=True,
                              target_means=[0., 0., 0., 0.],
                              target_stds=[0.1, 0.1, 0.2, 0.2])
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = torch.nn.modules.utils._pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = torch.nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            self.fc_cls = build_linear_layer(self.cls_predictor_cfg,
                                             in_features=in_channels,
                                             out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = build_linear_layer(self.reg_predictor_cfg,
                                             in_features=in_channels,
                                             out_features=out_dim_reg)

        assert (num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared conv and fcs
        self.shared_conv, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(self.num_shared_convs,
                                                                                     self.num_shared_fcs,
                                                                                     self.in_channels, True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_conv, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(self.num_cls_convs,
                                                                                  self.num_cls_fcs,
                                                                                  self.shared_out_channels)

        # add reg specific branch
        self.reg_conv, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(self.num_reg_convs,
                                                                                  self.num_reg_fcs,
                                                                                  self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = torch.nn.SiLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(self.cls_predictor_cfg,
                                             in_features=self.cls_last_dim,
                                             out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            self.fc_reg = build_linear_layer(self.reg_predictor_cfg,
                                             in_features=self.reg_last_dim,
                                             out_features=out_dim_reg)

    def _add_conv_fc_branch(self, num_branch_convs, num_branch_fcs, in_channels, is_shared=False):
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = torch.nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(ConvModule(conv_in_channels,
                                               self.conv_out_channels,
                                               3,
                                               padding=1,
                                               conv_cfg=self.conv_cfg,
                                               norm_cfg=self.norm_cfg,
                                               act_cfg=dict(type='SiLU')))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = torch.nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(torch.nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    @auto_fp16()
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_conv:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_conv:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_conv:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_boxes_list = [res.pos_bboxes for res in sampling_results]
        neg_boxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_boxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(self._get_target_single,
                                                                        pos_boxes_list,
                                                                        neg_boxes_list,
                                                                        pos_gt_boxes_list,
                                                                        pos_gt_labels_list,
                                                                        cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(cls_score,
                                          labels,
                                          label_weights,
                                          avg_factor=avg_factor,
                                          reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_indices = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_indices.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_indices.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[pos_indices.type(torch.bool),
                                                                             labels[pos_indices.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(pos_bbox_pred,
                                                     bbox_targets[pos_indices.type(torch.bool)],
                                                     bbox_weights[pos_indices.type(torch.bool)],
                                                     avg_factor=bbox_targets.size(0),
                                                     reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_indices].sum()
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = torch.nn.functional.softmax(cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            boxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            boxes = rois[:, 1:].clone()
            if img_shape is not None:
                boxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                boxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and boxes.size(0) > 0:
            scale_factor = boxes.new_tensor(scale_factor)
            boxes = (boxes.view(boxes.size(0), -1, 4) / scale_factor).view(boxes.size()[0], -1)

        if cfg is None:
            return boxes, scores
        else:
            det_boxes, det_labels = multiclass_nms(boxes, scores,
                                                   cfg.score_thr, cfg.nms,
                                                   cfg.max_per_img)

            return det_boxes, det_labels

    @force_fp32(apply_to=('bbox_preds',))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        boxes_list = []
        for i in range(len(img_metas)):
            indices = torch.nonzero(rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = indices.numel()

            boxes_ = rois[indices, 1:]
            label_ = labels[indices]
            bbox_pred_ = bbox_preds[indices]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(boxes_, label_, bbox_pred_, img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_indices = pos_is_gts_.new_ones(num_rois)
            keep_indices[:len(pos_is_gts_)] = pos_keep

            boxes_list.append(bboxes[keep_indices.type(torch.bool)])

        return boxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            indices = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, indices)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            boxes = self.bbox_coder.decode(rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], boxes), dim=1)

        return new_rois


@HEADS.register_module()
class Shared2FCBoxHead(BoxHead):
    def __init__(self, *args, **kwargs):
        super(Shared2FCBoxHead, self).__init__(num_shared_convs=0,
                                               num_shared_fcs=2,
                                               num_cls_convs=0,
                                               num_cls_fcs=0,
                                               num_reg_convs=0,
                                               num_reg_fcs=0,
                                               *args, **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBoxHead(BoxHead):
    def __init__(self, *args, **kwargs):
        super(Shared4Conv1FCBoxHead, self).__init__(num_shared_convs=4,
                                                    num_shared_fcs=1,
                                                    num_cls_convs=0,
                                                    num_cls_fcs=0,
                                                    num_reg_convs=0,
                                                    num_reg_fcs=0,
                                                    *args, **kwargs)


@HEADS.register_module()
class MaskHead(BaseModule):
    def __init__(self,
                 num_conv=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 up_cfg=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 predictor_cfg=None,
                 loss_mask=None):
        super().__init__()
        if loss_mask is None:
            loss_mask = dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        if predictor_cfg is None:
            predictor_cfg = dict(type='Conv')
        if up_cfg is None:
            up_cfg = dict(type='deconv', scale_factor=2)
        self.up_cfg = up_cfg.copy()
        self.num_conv = num_conv
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = torch.nn.modules.utils._pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.up_method = self.up_cfg.get('type')
        self.scale_factor = self.up_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.predictor_cfg = predictor_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        self.conv = ModuleList()
        for i in range(self.num_conv):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.conv.append(ConvModule(in_channels,
                                        self.conv_out_channels,
                                        self.conv_kernel_size,
                                        padding=padding,
                                        conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg,
                                        act_cfg=dict(type='SiLU')))
        up_in_channels = (self.conv_out_channels if self.num_conv > 0 else in_channels)
        up_cfg_copy = self.up_cfg.copy()
        if self.up_method is None:
            self.up = None
        elif self.up_method == 'deconv':
            up_cfg_copy.update(in_channels=up_in_channels,
                               out_channels=self.conv_out_channels,
                               kernel_size=self.scale_factor,
                               stride=self.scale_factor)
            self.up = build_upsample_layer(up_cfg_copy)
        elif self.up_method == 'carafe':
            up_cfg_copy.update(channels=up_in_channels, scale_factor=self.scale_factor)
            self.up = build_upsample_layer(up_cfg_copy)
        else:
            # suppress warnings
            align_corners = (None if self.up_method == 'nearest' else False)
            up_cfg_copy.update(scale_factor=self.scale_factor,
                               mode=self.up_method,
                               align_corners=align_corners)
            self.up = build_upsample_layer(up_cfg_copy)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (self.conv_out_channels
                             if self.up_method == 'deconv' else up_in_channels)
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1)
        self.relu = torch.nn.SiLU(inplace=True)

    @auto_fp16()
    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.up is not None:
            x = self.up(x)
            if self.up_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    @staticmethod
    def get_targets(sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_indices = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_indices, gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets, torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels,
                      rcnn_test_cfg, ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        # In most cases, scale_factor should have been
        # converted to Tensor when rescale the bbox
        if not isinstance(scale_factor, torch.Tensor):
            if isinstance(scale_factor, float):
                scale_factor = numpy.array([scale_factor] * 4)
                warnings.warn('Scale_factor should be a Tensor or ndarray '
                              'with shape (4,), float would be deprecated. ')
            assert isinstance(scale_factor, numpy.ndarray)
            scale_factor = torch.Tensor(scale_factor)

        if rescale:
            img_h, img_w = ori_shape[:2]
            bboxes = bboxes / scale_factor
        else:
            w_scale, h_scale = scale_factor[0], scale_factor[1]
            img_h = numpy.round(ori_shape[0] * h_scale.item()).astype(numpy.int32)
            img_w = numpy.round(ori_shape[1] * w_scale.item()).astype(numpy.int32)

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we neet to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(numpy.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(N, img_h, img_w,
                              device=device, dtype=torch.bool if threshold >= 0 else torch.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(mask_pred[inds],
                                                       bboxes[inds],
                                                       img_h,
                                                       img_w,
                                                       skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms


@HEADS.register_module()
class HTCMaskHead(MaskHead):
    def __init__(self, with_conv_res=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(self.conv_out_channels,
                                       self.conv_out_channels,
                                       1,
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=dict(type='SiLU'))

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.conv:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.up(x)
            if self.up_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]


@ROI_EXTRACTORS.register_module()
class RoIExtractor(BaseModule):
    def __init__(self, roi_layer, out_channels,
                 featmap_strides, finest_scale=56, global_context=False):
        super().__init__()
        self.finest_scale = finest_scale
        self.global_context = global_context
        self.pool = torch.nn.AdaptiveAvgPool2d(roi_layer.output_size)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        return len(self.featmap_strides)

    @staticmethod
    def build_roi_layers(layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = torch.nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    @staticmethod
    def roi_rescale(rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    def map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats',), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if self.global_context:
            global_context = []
            for feat in feats:
                global_context.append(self.pool(feat))
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        batch_size = feats[0].shape[0]
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            indices = mask.nonzero(as_tuple=False).squeeze(1)
            if indices.numel() > 0:
                rois_ = rois[indices]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[indices] = roi_feats_t
                if self.global_context:
                    for j in range(batch_size):
                        roi_feats_t[rois_[:, 0] == j] = roi_feats_t[rois_[:, 0] == j] + global_context[i][j]
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
