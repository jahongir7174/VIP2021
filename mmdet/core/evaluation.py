from collections.abc import Sequence
from multiprocessing import Pool

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable


def wider_face_classes():
    return ['face']


def voc_classes():
    return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def imagenet_det_classes():
    return ['accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
            'artichoke', 'axe', 'baby_bed', 'backpack', 'bagel', 'balance_beam',
            'banana', 'band_aid', 'banjo', 'baseball', 'basketball', 'bathing_cap',
            'beaker', 'bear', 'bee', 'bell_pepper', 'bench', 'bicycle', 'binder',
            'bird', 'bookshelf', 'bow_tie', 'bow', 'bowl', 'brassiere', 'burrito',
            'bus', 'butterfly', 'camel', 'can_opener', 'car', 'cart', 'cattle',
            'cello', 'centipede', 'chain_saw', 'chair', 'chime', 'cocktail_shaker',
            'coffee_maker', 'computer_keyboard', 'computer_mouse', 'corkscrew',
            'cream', 'croquet_ball', 'crutch', 'cucumber', 'cup_or_mug', 'diaper',
            'digital_clock', 'dishwasher', 'dog', 'domestic_cat', 'dragonfly',
            'drum', 'dumbbell', 'electric_fan', 'elephant', 'face_powder', 'fig',
            'filing_cabinet', 'flower_pot', 'flute', 'fox', 'french_horn', 'frog',
            'frying_pan', 'giant_panda', 'goldfish', 'golf_ball', 'golfcart',
            'guacamole', 'guitar', 'hair_dryer', 'hair_spray', 'hamburger',
            'hammer', 'hamster', 'harmonica', 'harp', 'hat_with_a_wide_brim',
            'head_cabbage', 'helmet', 'hippopotamus', 'horizontal_bar', 'horse',
            'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala_bear', 'ladle',
            'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
            'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk_can',
            'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck_brace',
            'oboe', 'orange', 'otter', 'pencil_box', 'pencil_sharpener', 'perfume',
            'person', 'piano', 'pineapple', 'ping-pong_ball', 'pitcher', 'pizza',
            'plastic_bag', 'plate_rack', 'pomegranate', 'popsicle', 'porcupine',
            'power_drill', 'pretzel', 'printer', 'puck', 'punching_bag', 'purse',
            'rabbit', 'racket', 'ray', 'red_panda', 'refrigerator',
            'remote_control', 'rubber_eraser', 'rugby_ball', 'ruler',
            'salt_or_pepper_shaker', 'saxophone', 'scorpion', 'screwdriver',
            'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
            'snowplow', 'soap_dispenser', 'soccer_ball', 'sofa', 'spatula',
            'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
            'strawberry', 'stretcher', 'sunglasses', 'swimming_trunks', 'swine',
            'syringe', 'table', 'tape_player', 'tennis_ball', 'tick', 'tie',
            'tiger', 'toaster', 'traffic_light', 'train', 'trombone', 'trumpet',
            'turtle', 'tv_or_monitor', 'unicycle', 'vacuum', 'violin',
            'volleyball', 'waffle_iron', 'washer', 'water_bottle', 'watercraft',
            'whale', 'wine_bottle', 'zebra'
            ]


def imagenet_vid_classes():
    return ['airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
            'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda',
            'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
            'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
            'watercraft', 'whale', 'zebra']


def coco_classes():
    return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


def cityscapes_classes():
    return ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


dataset_aliases = {'voc': ['voc', 'pascal_voc', 'voc07', 'voc12'],
                   'imagenet_det': ['det', 'imagenet_det', 'ilsvrc_det'],
                   'imagenet_vid': ['vid', 'imagenet_vid', 'ilsvrc_vid'],
                   'coco': ['coco', 'mscoco', 'ms_coco'],
                   'wider_face': ['WIDERFaceDataset', 'wider_face', 'WIDERFace'],
                   'cityscapes': ['cityscapes']}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if mmcv.is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def average_precision(recalls, precisions, mode='area'):
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_imagenet(det_bboxes,
                  gt_bboxes,
                  gt_bboxes_ignore=None,
                  default_iou_thr=0.5,
                  area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        default_iou_thr (float): IoU threshold to be considered as matched for
            medium and large bboxes (small ones have special rules).
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp
    # of a certain scale.
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
                    det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes - 1)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)),
                          default_iou_thr)
    # sort all detections by scores in descending order
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            # find best overlapped available gt
            for j in range(num_gts):
                # different from PASCAL VOC: allow finding other gts if the
                # best overlapped ones are already matched by other det bboxes
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            # there are 4 cases for a det bbox:
            # 1. it matches a gt, tp = 1, fp = 0
            # 2. it matches an ignored gt, tp = 0, fp = 0
            # 3. it matches no gt and within area range, tp = 0, fp = 1
            # 4. it matches no gt but is beyond area range, tp = 0, fp = 0
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate((np.zeros(gt_bboxes.shape[0], dtype=np.bool),
                                     np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (det_bboxes[:, 3] - det_bboxes[:, 1])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if min_area <= area < max_area:
                    fp[k, i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


def _recalls(all_ious, proposal_nums, thrs):
    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def set_recall_param(proposal_nums, iou_thrs):
    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, Sequence):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=0.5,
                 logger=None):
    img_num = len(gts)
    assert img_num == len(proposals)

    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)

    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4])
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)

    print_recall_summary(recalls, proposal_nums, iou_thrs, logger=logger)
    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)


def plot_num_recall(recalls, proposal_nums):
    """Plot Proposal_num-Recalls curve.

    Args:
        recalls(ndarray or list): shape (k,)
        proposal_nums(ndarray or list): same shape as `recalls`
    """
    if isinstance(proposal_nums, np.ndarray):
        _proposal_nums = proposal_nums.tolist()
    else:
        _proposal_nums = proposal_nums
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot([0] + _proposal_nums, [0] + _recalls)
    plt.xlabel('Proposal num')
    plt.ylabel('Recall')
    plt.axis([0, proposal_nums.max(), 0, 1])
    f.show()


def plot_iou_recall(recalls, iou_thrs):
    """Plot IoU-Recalls curve.

    Args:
        recalls(ndarray or list): shape (k,)
        iou_thrs(ndarray or list): same shape as `recalls`
    """
    if isinstance(iou_thrs, np.ndarray):
        _iou_thrs = iou_thrs.tolist()
    else:
        _iou_thrs = iou_thrs
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot(_iou_thrs + [1.0], _recalls + [0.])
    plt.xlabel('IoU')
    plt.ylabel('Recall')
    plt.axis([iou_thrs.min(), 1, 0, 1])
    f.show()


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             logger=None,
             tpfp_fn=None,
             nproc=4):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if tpfp_fn is None:
            if dataset in ['det', 'vid']:
                tpfp_fn = tpfp_imagenet
            else:
                tpfp_fn = tpfp_default
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_fn,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (
                        bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if dataset != 'voc07' else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
