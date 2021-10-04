import collections
import glob
import json
import os
import shutil

import cv2
import numpy
import tqdm
from pycocotools import mask as mask_util

data_dir = '../Dataset/Ins2021'
image_dir = 'images'
label_dir = 'labels'
crop_image_dir = 'c_images'
crop_label_dir = 'c_labels'
annotation_dir = 'annotation'
source_image_dir = 'src_images'
source_label_dir = 'src_labels'
aug_image_dir = 'aug_images'
aug_label_dir = 'aug_labels'


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mask_to_polygon(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = numpy.ascontiguousarray(mask)
    # some versions of cv2 does not support in contiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [(x + 0.5).tolist() for x in res if len(x) >= 6]
    return res, has_holes


def json2txt():
    print('Parsing ...')
    names = ['human', 'ball']
    json_files = glob.glob(f'{data_dir}/{annotation_dir}/*.json')
    segmentation = collections.defaultdict(list)

    for json_file in sorted(json_files):
        if 'test' in json_file:
            continue
        with open(json_file) as f:
            json_data = json.load(f)
        for annotation in json_data['annotations']:
            file_name = annotation['file_name']
            mask = mask_util.decode(annotation['segmentation'])
            polygons, has_hole = mask_to_polygon(mask)
            if len(polygons) > 0:
                seg_item = collections.defaultdict(list)
                for polygon in polygons:
                    poly = []
                    p = polygon
                    for i in range(0, len(p), 2):
                        poly.append([int(p[i]), int(p[i + 1])])
                    x, y, w, h = list(map(int, cv2.boundingRect(numpy.array([poly], int))))
                    area = w * h
                    seg_item[area].append(polygon)
                max_area = max(seg_item.keys())
                polys = seg_item[max_area]
                poly = []
                p = polys[0]
                if len(p):
                    for i in range(0, len(p), 2):
                        poly.append([int(p[i]), int(p[i + 1])])
                points = [names[annotation["category_id"] - 1]]
                for p in poly:
                    points.append(p[0])
                    points.append(p[1])
                points = " ".join([str(p) for p in points])
                segmentation[file_name].append(points)

    for key, value in segmentation.items():
        with open(f'{data_dir}/{label_dir}/{key[:-4]}.txt', 'w') as f:
            for v in value:
                f.write(f'{v}\n')

    for filename in [filename for filename in os.listdir(f'{data_dir}/train')]:
        shutil.copyfile(f'{data_dir}/train/{filename}', f'{data_dir}/{image_dir}/{filename}')
    for filename in [filename for filename in os.listdir(f'{data_dir}/val')]:
        shutil.copyfile(f'{data_dir}/val/{filename}', f'{data_dir}/{image_dir}/{filename}')


def crop_objects():
    print('Cropping ...')
    names = ['human', 'ball']
    counter = collections.defaultdict(int)
    name_set = collections.defaultdict(list)
    json_files = glob.glob(f'{data_dir}/{annotation_dir}/*.json')
    segmentation = collections.defaultdict(list)
    for json_file in sorted(json_files):
        if 'test' in json_file:
            continue
        with open(json_file) as f:
            json_data = json.load(f)
        for annotation in json_data['annotations']:
            file_name = annotation['file_name']
            img = cv2.imread(f'{data_dir}/{image_dir}/' + file_name)
            if not os.path.exists(f'{data_dir}/{crop_image_dir}/' + str(annotation['category_id'] - 1)):
                os.makedirs(f'{data_dir}/{crop_image_dir}/' + str(annotation['category_id'] - 1))
            mask = mask_util.decode(annotation['segmentation'])
            polygons, has_hole = mask_to_polygon(mask)
            if len(polygons) > 0:
                seg_item = collections.defaultdict(list)
                for polygon in polygons:
                    poly = []
                    p = polygon
                    for i in range(0, len(p), 2):
                        poly.append([int(p[i]), int(p[i + 1])])
                    x, y, w, h = list(map(int, cv2.boundingRect(numpy.array([poly], int))))
                    area = w * h
                    seg_item[area].append(polygon)
                max_area = max(seg_item.keys())
                polys = seg_item[max_area]
                poly = []
                p = polys[0]
                if len(p):
                    for i in range(0, len(p), 2):
                        poly.append([int(p[i]), int(p[i + 1])])

                rect = cv2.boundingRect(numpy.array([poly], int))  # returns (x,y,w,h) of the rect
                crop = img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                label = annotation["category_id"] - 1
                counter[label] += 1
                img_name = f'{names[label]}_{counter[label]}.png'
                points = [img_name]
                for p in poly:
                    points.append(p[0] - rect[0])
                    points.append(p[1] - rect[1])
                points = " ".join([str(p) for p in points])
                name_set[label].append(points)
                points = [names[label]]
                for p in poly:
                    points.append(p[0])
                    points.append(p[1])
                points = " ".join([str(p) for p in points])
                segmentation[file_name].append(points)
                cv2.imwrite(f'{data_dir}/{crop_image_dir}/{str(label)}/{img_name}', crop)

    if not os.path.exists(f'{data_dir}/{crop_label_dir}'):
        os.makedirs(f'{data_dir}/{crop_label_dir}')
    for key, value in name_set.items():
        with open(f'{data_dir}/{crop_label_dir}/{key}.txt', 'w') as f:
            for v in value:
                f.write(f'{v}\n')
    for key, value in segmentation.items():
        with open(f'{data_dir}/{label_dir}/{key[:-4]}.txt', 'w') as f:
            for v in value:
                f.write(f'{v}\n')


def sort_images():
    print('Sorting ...')
    label_names = [filename[:-4] for filename in os.listdir(f'{data_dir}/{label_dir}')]
    image_names = [filename[:-4] for filename in os.listdir(f'{data_dir}/{image_dir}')]
    for image_name in image_names:
        if image_name in label_names:
            with open(f'{data_dir}/{label_dir}/{image_name}.txt') as f:
                num_objects = len(f.readlines())
            if not os.path.exists(f'{data_dir}/{source_image_dir}/{str(num_objects)}'):
                os.makedirs(f'{data_dir}/{source_image_dir}/{str(num_objects)}')
            shutil.copyfile(f'{data_dir}/{image_dir}/{image_name}.png',
                            f'{data_dir}/{source_image_dir}/{str(num_objects)}/{image_name}.png')
        else:
            if not os.path.exists(f'{data_dir}/{source_image_dir}/0'):
                os.makedirs(f'{data_dir}/{source_image_dir}/0')
            shutil.copyfile(f'{data_dir}/{image_dir}/{image_name}.png',
                            f'{data_dir}/{source_image_dir}/0/{image_name}.png')


def copy_file():
    print('Copying ...')
    folders = [folder for folder in os.listdir(f'{data_dir}/{source_image_dir}')]
    for folder in folders:
        dst_names = [name[:-4] for name in os.listdir(f'{data_dir}/{source_image_dir}/{folder}')]
        if not os.path.exists(f'{data_dir}/{source_image_dir}/{folder}_aug'):
            os.makedirs(f'{data_dir}/{source_image_dir}/{folder}_aug')
            os.makedirs(f'{data_dir}/{source_label_dir}/{folder}_aug')
        for dst_name in dst_names:
            for i in range(1, 21, 1):
                shutil.copyfile(f'{data_dir}/{source_image_dir}/{folder}/{dst_name}.png',
                                f'{data_dir}/{source_image_dir}/{folder}_aug/{dst_name}:{i}.png')
                if str(folder) != '0':
                    shutil.copyfile(f'{data_dir}/{label_dir}/{dst_name}.txt',
                                    f'{data_dir}/{source_label_dir}/{folder}_aug/{dst_name}:{i}.txt')


def add_object_fn(dst_name):
    dst_img = cv2.imread(dst_name)

    if '0_aug' not in dst_name:
        dst_points = []
        filename = os.path.basename(dst_name).split(':')[0]
        with open(f'{data_dir}/{label_dir}/{filename}.txt') as f:
            for line in f.readlines():
                line = line.rstrip().split(' ')
                point = list(map(int, line[1:]))
                point.insert(0, line[0])
                point = " ".join([str(p) for p in point])
                dst_points.append(point)
    else:
        with open(f'{data_dir}/{crop_label_dir}/0.txt') as f:
            labels = {}
            for line in f.readlines():
                line = line.rstrip().split(' ')
                labels[line[0]] = line[1:]

        dst_h, dst_w = dst_img.shape[:2]

        poly = []
        dst_poly = []
        src_name = f'{data_dir}/{crop_image_dir}/0/human_1276.png'
        label = labels[os.path.basename(src_name)]
        src_img = cv2.imread(src_name)
        for i in range(0, len(label), 2):
            poly.append([int(label[i]), int(label[i + 1])])
        src_mask = numpy.zeros(src_img.shape, src_img.dtype)
        cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
        x_c, y_c = dst_w // 2, dst_h // 2
        for p in poly:
            dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
        dst_mask = numpy.zeros(dst_img.shape, dst_img.dtype)
        cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
        h, w = src_img.shape[:2]
        dst_points = []

        dst_point = ['human']
        for p in dst_poly:
            dst_point.append(p[0])
            dst_point.append(p[1])
        dst_point = " ".join([str(p) for p in dst_point])
        dst_points.append(dst_point)
        dst_img[dst_mask > 0] = 0
        dst_img[y_c:y_c + h, x_c:x_c + w] += src_img * (src_mask > 0)
    with open(f'{data_dir}/{aug_label_dir}/{os.path.basename(dst_name)[:-4]}.txt', 'w') as f:
        for dst_point in dst_points:
            f.write(f'{dst_point}\n')
    cv2.imwrite(f'{data_dir}/{aug_image_dir}/{os.path.basename(dst_name)}', dst_img)


def add_object():
    print('Augmenting ...')
    import multiprocessing
    folders = [folder for folder in os.listdir(f'{data_dir}/{source_image_dir}')]
    for folder in folders:
        if not folder.endswith('aug'):
            continue
        dst_names = glob.glob(f'{data_dir}/{source_image_dir}/{folder}/*.png')
        with multiprocessing.Pool(os.cpu_count() - 4) as pool:
            pool.map(add_object_fn, dst_names)
        pool.close()


def convert2coco():
    print('Converting into COCO ...')
    classes = ('human', 'ball')
    filenames = [filename for filename in os.listdir(f'{data_dir}/{aug_image_dir}')]
    img_id = 0
    box_id = 0
    images = []
    categories = []
    annotations = []
    for filename in tqdm.tqdm(filenames):
        img_id += 1
        h, w = cv2.imread(f"{data_dir}/{aug_image_dir}/{filename}").shape[:2]
        images.append({'file_name': filename, 'id': img_id, 'height': h, 'width': w})
        regions = []
        with open(f'{data_dir}/{aug_label_dir}/{filename[:-4]}.txt') as f:
            for line in f.readlines():
                regions.append(line.rstrip())
        for region in regions:
            box_id += 1
            region = region.split(' ')
            mask = region[1:]
            poly = []
            for i in range(0, len(mask), 2):
                poly.append([int(mask[i]), int(mask[i + 1])])
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([poly], int))
            bbox = [x_min, y_min, w, h]

            category_id = classes.index(region[0]) + 1
            annotations.append({'id': box_id,
                                'bbox': bbox,
                                'iscrowd': 0,
                                'image_id': img_id,
                                'segmentation': [list(map(int, mask))],
                                'area': bbox[2] * bbox[3],
                                'category_id': category_id})
    for category_id, category in enumerate(classes):
        categories.append({'supercategory': category, 'id': category_id + 1, 'name': category})
    print(len(images), 'images')
    print(len(annotations), 'instances')
    json_data = json.dumps({'images': images, 'categories': categories, 'annotations': annotations})
    with open(f'{data_dir}/{annotation_dir}/train_aug.json', 'w') as f:
        f.write(json_data)


def main():
    make_dir(os.path.join(data_dir, image_dir))
    make_dir(os.path.join(data_dir, label_dir))
    make_dir(os.path.join(data_dir, crop_image_dir))
    make_dir(os.path.join(data_dir, crop_label_dir))
    make_dir(os.path.join(data_dir, source_image_dir))
    make_dir(os.path.join(data_dir, source_label_dir))
    make_dir(os.path.join(data_dir, aug_image_dir))
    make_dir(os.path.join(data_dir, aug_label_dir))
    json2txt()
    crop_objects()
    sort_images()
    copy_file()
    add_object()
    convert2coco()


if __name__ == '__main__':
    main()
