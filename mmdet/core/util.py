import warnings
import numpy as np
import torch
import os


def get_k_for_topk(k, size):
    ret_k = -1
    if k <= 0 or size <= 0:
        return ret_k
    if torch.onnx.is_in_onnx_export():
        is_trt_backend = os.environ.get('ONNX_BACKEND') == 'MMCVTensorRT'
        if is_trt_backend:
            # TensorRT does not support dynamic K with TopK op
            if 0 < k < size:
                ret_k = k
        else:
            # Always keep topk op for dynamic input in onnx for ONNX Runtime
            ret_k = torch.where(k < size, k, size)
    elif k < size:
        ret_k = k
    else:
        # ret_k is -1
        pass
    return ret_k


def bbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def images_to_levels(target, num_levels):
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
                       (flat_anchors[:, 0] >= -allowed_border) & \
                       (flat_anchors[:, 1] >= -allowed_border) & \
                       (flat_anchors[:, 2] < img_w + allowed_border) & \
                       (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def ensure_rng(rng=None):
    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


def random_boxes(num=1, scale=1, rng=None):
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes


class NiceRepr:
    def __nice__(self):
        if hasattr(self, '__len__'):
            return str(len(self))
        else:
            raise NotImplementedError(f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)
