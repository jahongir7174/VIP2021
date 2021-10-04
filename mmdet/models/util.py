import asyncio
import contextlib
import functools
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
import typing

import apex
import mmcv
import torch
from mmcv.ops.nms import batched_nms
from mmcv.parallel import is_module_wrapper
from mmcv.runner import OptimizerHook, HOOKS, RUNNERS, EpochBasedRunner, OPTIMIZERS
from mmcv.runner.checkpoint import weights_to_cpu, get_state_dict
from six.moves import map, zip

from .builder import LOSSES

logger = logging.getLogger(__name__)

DEBUG_COMPLETED_TIME = bool(os.environ.get('DEBUG_COMPLETED_TIME', False))


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], keep
    else:
        return dets, labels[keep]


def multi_apply(func, *args, **kwargs):
    p_func = functools.partial(func, **kwargs) if kwargs else func
    map_results = map(p_func, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


if sys.version_info >= (3, 7):
    @contextlib.asynccontextmanager
    async def completed(trace_name='',
                        name='',
                        sleep_interval=0.05,
                        streams: typing.List[torch.cuda.Stream] = None):
        if not torch.cuda.is_available():
            yield
            return

        stream_before_context_switch = torch.cuda.current_stream()
        if not streams:
            streams = [stream_before_context_switch]
        else:
            streams = [s if s else stream_before_context_switch for s in streams]

        end_events = [torch.cuda.Event(enable_timing=DEBUG_COMPLETED_TIME) for _ in streams]

        if DEBUG_COMPLETED_TIME:
            start = torch.cuda.Event(enable_timing=True)
            stream_before_context_switch.record_event(start)

            cpu_start = time.monotonic()
        logger.debug('%s %s starting, streams: %s', trace_name, name, streams)
        grad_enabled_before = torch.is_grad_enabled()
        try:
            yield
        finally:
            current_stream = torch.cuda.current_stream()
            assert current_stream == stream_before_context_switch

            if DEBUG_COMPLETED_TIME:
                cpu_end = time.monotonic()
            for i, stream in enumerate(streams):
                event = end_events[i]
                stream.record_event(event)

            grad_enabled_after = torch.is_grad_enabled()

            # observed change of torch.is_grad_enabled() during concurrent run of
            # async_test_bboxes code
            assert (grad_enabled_before == grad_enabled_after), 'Unexpected is_grad_enabled() value change'

            are_done = [e.query() for e in end_events]
            logger.debug('%s %s completed: %s streams: %s', trace_name, name, are_done, streams)
            with torch.cuda.stream(stream_before_context_switch):
                while not all(are_done):
                    await asyncio.sleep(sleep_interval)
                    are_done = [e.query() for e in end_events]
                    logger.debug('%s %s completed: %s streams: %s',
                                 trace_name,
                                 name,
                                 are_done,
                                 streams)

            current_stream = torch.cuda.current_stream()
            assert current_stream == stream_before_context_switch

            if DEBUG_COMPLETED_TIME:
                cpu_time = (cpu_end - cpu_start) * 1000
                stream_times_ms = ''
                for i, stream in enumerate(streams):
                    elapsed_time = start.elapsed_time(end_events[i])
                    stream_times_ms += f' {stream} {elapsed_time:.2f} ms'
                logger.info('%s %s %.2f ms %s', trace_name, name, cpu_time, stream_times_ms)


def reduce_loss(loss, reduction):
    reduction_enum = torch.nn.functional._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


@mmcv.jit(derivate=True, coderize=True)
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


@mmcv.jit(coderize=True)
def accuracy(pred, target, topk=1, thresh=None):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    # element-wise losses
    loss = torch.nn.functional.cross_entropy(pred, label,
                                             weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    indices = torch.nonzero((labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if indices.numel() > 0:
        bin_labels[indices, labels[indices]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    if pred.dim() != label.dim():
        label, weight = _expand_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label.float(),
                                                                pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    indices = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[indices, label].squeeze(1)
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_slice, target,
                                                                weight=class_weight, reduction='mean')[None]


def seesaw_ce_loss(cls_score,
                   labels,
                   label_weights,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   eps,
                   reduction='mean',
                   avg_factor=None):
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(one_hot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = torch.nn.functional.softmax(cls_score.detach(), dim=1)
        self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - one_hot_labels))

    loss = torch.nn.functional.cross_entropy(cls_score, labels,
                                             weight=None, reduction='none')

    if label_weights is not None:
        label_weights = label_weights.float()
    loss = weight_reduce_loss(loss, weight=label_weights, reduction=reduction, avg_factor=avg_factor)
    return loss


def save_checkpoint(model, filename, optimizer=None, meta=None):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {'meta': meta,
                  'state_dict': weights_to_cpu(get_state_dict(model))}
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, torch.optim.Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save amp state dict in the checkpoint
    checkpoint['amp'] = apex.amp.state_dict()

    if filename.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError('Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = os.path.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_file = os.path.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(os.path.dirname(filename))
        # immediately flush buffer
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()


@LOSSES.register_module()
class CrossEntropyLoss(torch.nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(cls_score,
                                                         label,
                                                         weight,
                                                         class_weight=class_weight,
                                                         reduction=reduction,
                                                         avg_factor=avg_factor,
                                                         **kwargs)
        return loss_cls


@LOSSES.register_module()
class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(pred,
                                                      target,
                                                      weight,
                                                      beta=self.beta,
                                                      reduction=reduction,
                                                      avg_factor=avg_factor,
                                                      **kwargs)
        return loss_bbox


@LOSSES.register_module()
class SeesawLoss(torch.nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1203,
                 eps=1e-2,
                 reduction='mean',
                 loss_weight=1.0,
                 return_dict=True):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_dict = return_dict

        # 0 for pos, 1 for neg
        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        self.register_buffer('cum_samples',
                             torch.zeros(self.num_classes + 1, dtype=torch.float))

        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

    def _split_cls_score(self, cls_score):
        # split cls_score to cls_score_classes and cls_score_objectness
        assert cls_score.size(-1) == self.num_classes + 2
        cls_score_classes = cls_score[..., :-2]
        cls_score_objectness = cls_score[..., -2:]
        return cls_score_classes, cls_score_objectness

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        return num_classes + 2

    def get_activation(self, cls_score):
        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)
        score_classes = torch.nn.functional.softmax(cls_score_classes, dim=-1)
        score_objectness = torch.nn.functional.softmax(cls_score_objectness, dim=-1)
        score_pos = score_objectness[..., [0]]
        score_neg = score_objectness[..., [1]]
        score_classes = score_classes * score_pos
        scores = torch.cat([score_classes, score_neg], dim=-1)
        return scores

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        obj_labels = (labels == self.num_classes).long()
        cls_score_classes, cls_score_objectness = self._split_cls_score(cls_score)
        acc_objectness = accuracy(cls_score_objectness, obj_labels)
        acc_classes = accuracy(cls_score_classes[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_objectness'] = acc_objectness
        acc['acc_classes'] = acc_classes
        return acc

    def forward(self,
                cls_score,
                labels,
                label_weights=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes + 2
        pos_inds = labels < self.num_classes
        # 0 for pos, 1 for neg
        obj_labels = (labels == self.num_classes).long()

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if label_weights is not None:
            label_weights = label_weights.float()
        else:
            label_weights = labels.new_ones(labels.size(), dtype=torch.float)

        cls_score_classes, cls_score_objectness = self._split_cls_score(
            cls_score)
        # calculate loss_cls_classes (only need pos samples)
        if pos_inds.sum() > 0:
            loss_cls_classes = self.loss_weight * self.cls_criterion(cls_score_classes[pos_inds],
                                                                     labels[pos_inds],
                                                                     label_weights[pos_inds],
                                                                     self.cum_samples[:self.num_classes],
                                                                     self.num_classes, self.p,
                                                                     self.q, self.eps,
                                                                     reduction,
                                                                     avg_factor)
        else:
            loss_cls_classes = cls_score_classes[pos_inds].sum()
        # calculate loss_cls_objectness
        loss_cls_objectness = self.loss_weight * cross_entropy(cls_score_objectness,
                                                               obj_labels,
                                                               label_weights,
                                                               reduction,
                                                               avg_factor)

        if self.return_dict:
            loss_cls = dict()
            loss_cls['loss_cls_objectness'] = loss_cls_objectness
            loss_cls['loss_cls_classes'] = loss_cls_classes
        else:
            loss_cls = loss_cls_classes + loss_cls_objectness
        return loss_cls


@RUNNERS.register_module()
class EpochBasedRunnerAMP(EpochBasedRunner):
    def save_checkpoint(self, out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True, meta=None, create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = os.path.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = os.path.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def resume(self, checkpoint,
               resume_optimizer=True, map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(checkpoint,
                                                  map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, torch.optim.Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(checkpoint['optimizer'][k])
            else:
                raise TypeError('Optimizer should be dict or torch.optim.Optimizer '
                                f'but got {type(self.optimizer)}')

        if 'amp' in checkpoint:
            apex.amp.load_state_dict(checkpoint['amp'])
            self.logger.info('load amp state dict')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)


@HOOKS.register_module()
class DistributedOptimizerHook(OptimizerHook):
    def __init__(self, update_interval=1, grad_clip=None, coalesce=True, bucket_size_mb=-1, use_fp16=False):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        runner.outputs['loss'] /= self.update_interval
        if self.use_fp16:
            with apex.amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            runner.outputs['loss'].backward()
        if self.every_n_iters(runner, self.update_interval):
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()
            runner.optimizer.zero_grad()


@OPTIMIZERS.register_module()
class CenterGradSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, use_gc=False, gc_conv_only=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, use_gc=use_gc, gc_conv_only=gc_conv_only)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @staticmethod
    def center_grad(x, use_gc=True, gc_conv_only=False):
        if use_gc:
            if gc_conv_only:
                if len(list(x.size())) > 3:
                    x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
            else:
                if len(list(x.size())) > 1:
                    x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        return x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                d_p = self.center_grad(d_p, use_gc=group['use_gc'],
                                       gc_conv_only=group['gc_conv_only'])

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
