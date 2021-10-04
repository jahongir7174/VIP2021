import collections
import os
import random
import warnings

import cv2
import mmcv
import numpy
from PIL import Image, ImageOps, ImageEnhance
from mmcv.parallel import DataContainer
from mmcv.utils import build_from_cfg
from pycocotools import mask as mask_util

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.util import to_tensor
from .builder import PIPELINES


def bbox2fields():
    bbox2label = {'gt_bboxes': 'gt_labels', 'gt_bboxes_ignore': 'gt_labels_ignore'}
    bbox2mask = {'gt_bboxes': 'gt_masks', 'gt_bboxes_ignore': 'gt_masks_ignore'}
    return bbox2label, bbox2mask


def invert(image, _):
    return ImageOps.invert(image)


def equalize(image, _):
    return ImageOps.equalize(image)


def solar1(image, magnitude):
    return ImageOps.solarize(image, int((magnitude / 10.) * 256))


def solar2(image, magnitude):
    return ImageOps.solarize(image, 256 - int((magnitude / 10.) * 256))


def solar3(image, magnitude):
    lut = []
    for i in range(256):
        if i < 128:
            lut.append(min(255, i + int((magnitude / 10.) * 110)))
        else:
            lut.append(i)
    if image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        return image


def poster1(image, magnitude):
    magnitude = int((magnitude / 10.) * 4)
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def poster2(image, magnitude):
    magnitude = 4 - int((magnitude / 10.) * 4)
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def poster3(image, magnitude):
    magnitude = int((magnitude / 10.) * 4) + 4
    if magnitude >= 8:
        return image
    return ImageOps.posterize(image, magnitude)


def contrast1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude
    return ImageEnhance.Contrast(image).enhance(magnitude)


def contrast2(image, magnitude):
    return ImageEnhance.Contrast(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def contrast3(image, _):
    return ImageOps.autocontrast(image)


def color1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude
    return ImageEnhance.Color(image).enhance(magnitude)


def color2(image, magnitude):
    return ImageEnhance.Color(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def brightness1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude
    return ImageEnhance.Brightness(image).enhance(magnitude)


def brightness2(image, magnitude):
    return ImageEnhance.Brightness(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def sharpness1(image, magnitude):
    magnitude = (magnitude / 10.) * .9
    magnitude = 1.0 + -magnitude if random.random() > 0.5 else magnitude

    return ImageEnhance.Sharpness(image).enhance(magnitude)


def sharpness2(image, magnitude):
    return ImageEnhance.Sharpness(image).enhance((magnitude / 10.) * 1.8 + 0.1)


def random_hsv(img, h_gain=0.015, s_gain=0.700, v_gain=0.400):
    random_gain = numpy.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=numpy.int16)
    lut_hue = ((x * random_gain[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * random_gain[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * random_gain[2], 0, 255).astype('uint8')

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype('uint8')
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


class Shear:
    def __init__(self, min_val=0.002, max_val=0.2):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _shear_img(results, magnitude, direction):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_sheared = mmcv.imshear(img,
                                       magnitude,
                                       direction)
            results[key] = img_sheared.astype(img.dtype)

    @staticmethod
    def _shear_boxes(results, magnitude, direction):
        h, w, c = results['img_shape']
        if direction == 'horizontal':
            shear_matrix = numpy.stack([[1, magnitude], [0, 1]]).astype(numpy.float32)  # [2, 2]
        else:
            shear_matrix = numpy.stack([[1, 0], [magnitude, 1]]).astype(numpy.float32)
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])  # [4, 2, nb_box, 1]
            coordinates = coordinates[..., 0].transpose((2, 1, 0)).astype(numpy.float32)  # [nb_box, 2, 4]
            new_coords = numpy.matmul(shear_matrix[None, :, :], coordinates)  # [nb_box, 2, 4]
            min_x = numpy.min(new_coords[:, 0, :], axis=-1)
            min_y = numpy.min(new_coords[:, 1, :], axis=-1)
            max_x = numpy.max(new_coords[:, 0, :], axis=-1)
            max_y = numpy.max(new_coords[:, 1, :], axis=-1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _shear_masks(results, magnitude, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.shear((h, w),
                                       magnitude,
                                       direction)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            return results
        magnitude = numpy.random.uniform(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            magnitude *= -1
        direction = numpy.random.choice(['horizontal', 'vertical'])
        self._shear_img(results, magnitude, direction)
        self._shear_boxes(results, magnitude, direction)
        self._shear_masks(results, magnitude, direction)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


class Rotate:
    def __init__(self, min_val=1, max_val=45):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _rotate_img(results, angle, center, scale):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(img, angle, center, scale)
            results[key] = img_rotated.astype(img.dtype)

    @staticmethod
    def _rotate_boxes(results, rotate_matrix):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(
                results[key], results[key].shape[-1], axis=-1)
            coordinates = numpy.stack([[min_x, min_y],
                                       [max_x, min_y],
                                       [min_x, max_y],
                                       [max_x, max_y]])
            coordinates = numpy.concatenate((coordinates,
                                             numpy.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype)),
                                            axis=1)
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = numpy.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x = numpy.min(rotated_coords[:, :, 0], axis=1)
            min_y = numpy.min(rotated_coords[:, :, 1], axis=1)
            max_x = numpy.max(rotated_coords[:, :, 0], axis=1)
            max_y = numpy.max(rotated_coords[:, :, 1], axis=1)
            min_x = numpy.clip(min_x, a_min=0, a_max=w)
            min_y = numpy.clip(min_y, a_min=0, a_max=h)
            max_x = numpy.clip(max_x, a_min=min_x, a_max=w)
            max_y = numpy.clip(max_y, a_min=min_y, a_max=h)
            results[key] = numpy.stack([min_x, min_y, max_x, max_y], axis=-1).astype(results[key].dtype)

    @staticmethod
    def _rotate_masks(results, angle, center, scale):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, 0)

    @staticmethod
    def _filter_invalid(results, min_bbox_size=0):
        bbox2label, bbox2mask = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > min_bbox_size) & (bbox_h > min_bbox_size)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            return results
        h, w = results['img'].shape[:2]
        angle = numpy.random.randint(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            angle *= -1
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1)
        self._rotate_img(results, angle, center, 1)
        self._rotate_boxes(results, matrix)
        self._rotate_masks(results, angle, center, 1)
        self._filter_invalid(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


class Translate:
    def __init__(self, min_val=1, max_val=256):
        self.min_val = min_val
        self.max_val = max_val

    @staticmethod
    def _translate_image(results, offset, direction):
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(img, offset, direction).astype(img.dtype)

    @staticmethod
    def _translate_boxes(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('bbox_fields', []):
            min_x, min_y, max_x, max_y = numpy.split(results[key], results[key].shape[-1], axis=-1)
            if direction == 'horizontal':
                min_x = numpy.maximum(0, min_x + offset)
                max_x = numpy.minimum(w, max_x + offset)
            elif direction == 'vertical':
                min_y = numpy.maximum(0, min_y + offset)
                max_y = numpy.minimum(h, max_y + offset)

            results[key] = numpy.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    @staticmethod
    def _translate_masks(results, offset, direction):
        h, w, c = results['img_shape']
        for key in results.get('mask_fields', []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, 0)

    @staticmethod
    def _filter_invalid(results):
        bbox2label, bbox2mask = bbox2fields()
        for key in results.get('bbox_fields', []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_indices = (bbox_w > 0) & (bbox_h > 0)
            valid_indices = numpy.nonzero(valid_indices)[0]
            results[key] = results[key][valid_indices]
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_indices]
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_indices]
        return results

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            return results
        offset = numpy.random.randint(self.min_val, self.max_val)
        if numpy.random.rand() > 0.5:
            offset *= -1
        direction = numpy.random.choice(['horizontal', 'vertical'])
        self._translate_image(results, offset, direction)
        self._translate_boxes(results, offset, direction)
        self._translate_masks(results, offset, direction)
        self._filter_invalid(results)
        return results


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadImageFromFile:
    def __init__(self,
                 to_float32=False,
                 color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = dict(backend='disk').copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = os.path.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(numpy.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class LoadAnnotations:
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 copy_paste=True,
                 data_dir='../Dataset/Ins2021'):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_label = with_label
        self.file_client_args = dict(backend='disk').copy()
        self.file_client = None

        # CopyPaste
        if copy_paste:
            self.data_dir = data_dir
            self.copy_paste = copy_paste

            self.crop_image_dir = 'c_images'
            self.crop_label_dir = 'c_labels'

            import glob
            self.src_names_0 = glob.glob(f'{data_dir}/{self.crop_image_dir}/0/*.png')
            self.src_names_1 = glob.glob(f'{data_dir}/{self.crop_image_dir}/1/*.png')

            with open(f'{data_dir}/{self.crop_label_dir}/0.txt') as f:
                self.labels = {}
                for line in f.readlines():
                    line = line.rstrip().split(' ')
                    self.labels[line[0]] = line[1:]
            with open(f'{data_dir}/{self.crop_label_dir}/1.txt') as f:
                for line in f.readlines():
                    line = line.rstrip().split(' ')
                    self.labels[line[0]] = line[1:]

    @staticmethod
    def _load_boxes(results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_boxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_boxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_boxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    @staticmethod
    def _load_labels(results):
        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        if self.copy_paste:
            img = results['img']
            img, label, boxes, masks = self.add_objects(img)

            masks.extend(results['ann_info']['masks'])
            label.extend(results['gt_labels'].tolist())
            boxes.extend(results['gt_bboxes'].tolist())

            results['img'] = img
            results['gt_labels'] = numpy.array(label, numpy.int64)
            results['gt_bboxes'] = numpy.array(boxes, numpy.float32)
        else:
            masks = results['ann_info']['masks']
        gt_masks = BitmapMasks([self._poly2mask(mask, h, w) for mask in masks], h, w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    @staticmethod
    def _poly2mask(mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rle = mask_util.merge(mask_util.frPyObjects(mask_ann, img_h, img_w))
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = mask_util.decode(rle)
        return mask

    def add_objects(self, img):

        gt_label = []
        gt_masks = []
        gt_boxes = []

        dst_h, dst_w = img.shape[:2]
        num_0 = numpy.random.randint(5, 15)
        num_1 = numpy.random.randint(5, 15)
        y_c_list = numpy.random.randint(dst_h // 2 - 256, dst_h // 2 + 256, num_0 + num_1)
        x_c_list = numpy.random.randint(256, dst_w - 256, num_0 + num_1)
        src_names = numpy.random.choice(self.src_names_0, num_0).tolist()
        src_names.extend(numpy.random.choice(self.src_names_1, num_1).tolist())

        mask_list = []
        poly_list = []
        src_img_list = []
        src_name_list = []
        for src_name in src_names:
            poly = []
            label = self.labels[os.path.basename(src_name)]
            src_img = cv2.imread(src_name)
            for i in range(0, len(label), 2):
                poly.append([int(label[i]), int(label[i + 1])])
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            mask_list.append(src_mask)
            poly_list.append(poly)
            src_img_list.append(src_img)
            src_name_list.append(src_name)
        for i, (x_c, y_c) in enumerate(zip(x_c_list, y_c_list)):
            dst_poly = []
            for p in poly_list[i]:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = numpy.zeros(img.shape, img.dtype)
            cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([dst_poly], int))
            gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
            src = src_img_list[i].copy()
            h, w = src.shape[:2]
            mask = mask_list[i].copy()
            img[dst_mask > 0] = 0
            img[y_c:y_c + h, x_c:x_c + w] += src * (mask > 0)
            if 'human' in os.path.basename(src_name_list[i]):
                gt_label.append(0)
            else:
                gt_label.append(1)
            dst_point = []
            for p in dst_poly:
                dst_point.append(p[0])
                dst_point.append(p[1])
            gt_masks.append([dst_point])
        return img, gt_label, gt_boxes, gt_masks

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_boxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class RandomAugment:
    def __init__(self):
        self.color_transforms = [color1, color2, solar1, solar2, solar3, invert,
                                 poster1, poster2, poster3, equalize, contrast1,
                                 contrast2, contrast3, sharpness1, sharpness2,
                                 brightness1, brightness2]
        self.geo_transforms = [Shear(), Rotate(), Translate()]

    def __call__(self, results):
        if numpy.random.rand() > 0.5:
            random_hsv(results['img'])
        else:
            image = results['img']
            image = Image.fromarray(image[:, :, ::-1])
            for transform in numpy.random.choice(self.color_transforms, 2):
                magnitude = min(10., max(0., random.gauss(9, 0.5)))
                image = transform(image, magnitude)
            results['img'] = numpy.array(image)[:, :, ::-1]

        transform = numpy.random.choice(self.geo_transforms)
        return transform(results)

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Resize:
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = numpy.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = numpy.random.randint(min(img_scale_long),
                                         max(img_scale_long) + 1)
        short_edge = numpy.random.randint(min(img_scale_short),
                                          max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = numpy.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_image(self, results):
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(results[key],
                                                   results['scale'],
                                                   return_scale=True,
                                                   backend=self.backend)
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(results[key],
                                                      results['scale'],
                                                      return_scale=True,
                                                      backend=self.backend)
            results[key] = img

            scale_factor = numpy.array([w_scale, h_scale, w_scale, h_scale], dtype=numpy.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_boxes(self, results):
        for key in results.get('bbox_fields', []):
            boxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                boxes[:, 0::2] = numpy.clip(boxes[:, 0::2], 0, img_shape[1])
                boxes[:, 1::2] = numpy.clip(boxes[:, 1::2], 0, img_shape[0])
            results[key] = boxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def __call__(self, results):
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, 'scale and scale_factor cannot be both set.'
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_image(results)
        self._resize_boxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class RandomFlip:
    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert mmcv.is_list_of(flip_ratio, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]

            cur_dir = numpy.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Pad:
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(results[key],
                                        shape=self.size,
                                        pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(results[key],
                                                    self.size_divisor,
                                                    pad_val=self.pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Normalize:
    def __init__(self, mean, std, to_rgb=True):
        self.mean = numpy.array(mean, dtype=numpy.float32)
        self.std = numpy.array(std, dtype=numpy.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key],
                                            self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class GridMask:
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.5):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def __call__(self, results):
        img = results['img']
        if numpy.random.rand() > self.prob:
            return results
        h = img.shape[0]
        w = img.shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = numpy.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = numpy.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = numpy.ones((hh, ww), numpy.float32)
        st_h = numpy.random.randint(d)
        st_w = numpy.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = numpy.random.randint(self.rotate)
        mask = Image.fromarray(numpy.uint8(mask))
        mask = mask.rotate(r)
        mask = numpy.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(numpy.float32)
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(img)
        mask = numpy.expand_dims(mask, 2).repeat(3, axis=2)
        if self.offset:
            offset = 2 * (numpy.random.rand(h, w) - 0.5)
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        results['img'] = img
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class MultiScaleFlipAug:
    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), 'Must have but only one variable can be setted'
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn('flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn('flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction) for direction in self.flip_direction]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ImageToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = numpy.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class DefaultFormatBundle:
    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = numpy.expand_dims(img, -1)
            img = numpy.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DataContainer(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DataContainer(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DataContainer(results['gt_masks'], cpu_only=True)
        return results

    def _add_default_meta_keys(self, results):
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault('img_norm_cfg',
                           dict(mean=numpy.zeros(num_channels, dtype=numpy.float32),
                                std=numpy.ones(num_channels, dtype=numpy.float32),
                                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect:
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DataContainer(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__
