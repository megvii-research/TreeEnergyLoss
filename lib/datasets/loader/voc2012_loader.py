from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pdb
import zipfile
from xml.etree import ElementTree

import numpy as np
from torch.utils import data
from torchvision.datasets import VOCSegmentation, SBDataset

from lib.utils.helpers.image_helper import ImageHelper
from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log

from PIL import Image, ImageDraw
import tqdm

MOD_ID = 'id'
MOD_RGB = 'rgb'
MOD_SS_DENSE = 'semseg_dense'
MOD_SS_CLICKS = 'semseg_clicks'
MOD_SS_SCRIBBLES = 'semseg_scribbles'
MOD_VALIDITY = 'validity_mask'

SPLIT_TRAIN = 'train'
SPLIT_VALID = 'val'

MODE_INTERP = {
    MOD_ID: None,
    MOD_RGB: 'bilinear',
    MOD_SS_DENSE: 'nearest',
    MOD_SS_CLICKS:  'sparse',
    MOD_SS_SCRIBBLES: 'sparse',
    MOD_VALIDITY: 'nearest',
}


class VOC2012Loader(data.Dataset):
    def __init__(self, root_dir, split, aug_transform=None, dataset=None,
                 img_transform=None, label_transform=None, configer=None):
        self.split = split
        self.configer = configer
        self.aug_transform = aug_transform
        self.img_transform = img_transform
        self.label_transform = label_transform

        size_mode = self.configer.get(dataset, 'data_transformer')['size_mode']
        self.is_stack = size_mode != 'diverse_size'

        root_voc = os.path.join(root_dir, 'VOC')
        root_sbd = os.path.join(root_dir, 'SBD')

        download = False
        if not os.path.exists(root_voc):
            download = True

        self.ds_voc_valid = VOCSegmentation(root_voc, image_set=SPLIT_VALID, download=download)

        if split == SPLIT_TRAIN:
            self.ds_voc_train = VOCSegmentation(root_voc, image_set=SPLIT_TRAIN, download=download)
            self.ds_sbd_train = SBDataset(
                root_sbd,
                image_set=SPLIT_TRAIN,
                download=download
            )
            self.ds_sbd_valid = SBDataset(root_sbd, image_set=SPLIT_VALID, download=download)

            self.name_to_ds_id = {
                self._sample_name(path): (self.ds_sbd_train, i) for i, path in enumerate(self.ds_sbd_train.images)
            }
            self.name_to_ds_id.update({
                self._sample_name(path): (self.ds_sbd_valid, i) for i, path in enumerate(self.ds_sbd_valid.images)
            })
            self.name_to_ds_id.update({
                self._sample_name(path): (self.ds_voc_train, i) for i, path in enumerate(self.ds_voc_train.images)
            })
            for path in self.ds_voc_valid.images:
                name = self._sample_name(path)
                self.name_to_ds_id.pop(name, None)
        else:
            self.name_to_ds_id = {
                self._sample_name(path): (self.ds_voc_valid, i) for i, path in enumerate(self.ds_voc_valid.images)
            }

        self.sample_names = list(sorted(self.name_to_ds_id.keys()))
        self.transforms = None

        path_points_fg = os.path.join(root_dir, 'voc_whats_the_point.json')
        path_points_bg = os.path.join(root_dir, 'voc_whats_the_point_bg_from_scribbles.json')
        with open(path_points_fg, 'r') as f:
            self.ds_clicks_fg = json.load(f)
        with open(path_points_bg, 'r') as f:
            self.ds_clicks_bg = json.load(f)
        self.ds_scribbles_path = os.path.join(root_dir, 'voc_scribbles.zip')
        assert os.path.isfile(self.ds_scribbles_path), f'Scribbles not found at {self.ds_scribbles_path}'
        self.cls_name_to_id = {name: i for i, name in enumerate(self.semseg_class_names)}
        self._semseg_class_histogram = self._compute_histogram()

        self.integrity_check = True
        self.stroke_width = 3
        self.semseg_ignore_class = 255

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return len(self.sample_names)

    def get(self, index):
        ds, idx = self.name_to_ds_id[self.name_from_index(index)]

        name = self._sample_name(ds.images[idx])
        img = ImageHelper.read_image(ds.images[idx],
                                     tool=self.configer.get('data', 'image_tool'),
                                     mode=self.configer.get('data', 'input_mode'))

        width, height = ImageHelper.get_size(img)
        img_size = (width, height)
        ss_dense_path = ds.masks[idx]

        if ss_dense_path.endswith('mat'):
            ss_dense = ds._get_segmentation_target(ss_dense_path)
        else:
            ss_dense = Image.open(ss_dense_path)

        assert not self.integrity_check or ss_dense.size == img_size, \
            f'RGB and SEMSEG shapes do not match in sample {name}'

        if self.split == "train":
            annotation = self.configer.get('data', 'annotation')
            if annotation == 'point':
                ss_clicks_fg = self.ds_clicks_fg[name]
                ss_clicks_bg = self.ds_clicks_bg[name]
                ss_clicks = ss_clicks_fg + ss_clicks_bg
                ss_clicks = [(d['cls'], [(d['x'], d['y'])]) for d in ss_clicks]
                ss_clicks = self.rasterize_clicks(ss_clicks, width, height)
                labelmap = np.array(ss_clicks)
            elif annotation == 'scribble':
                ss_scribbles = self._parse_scribble(
                    name,
                    known_width=width if self.integrity_check else None,
                    known_height=height if self.integrity_check else None
                )
                ss_scribbles = self.rasterize_scribbles(ss_scribbles, width, height)
                labelmap = np.array(ss_scribbles)
            elif annotation == 'mask':
                labelmap = np.array(ss_dense)
            else:
                raise NotImplementedError

        elif self.split == "val":
            labelmap = np.array(ss_dense)
        else:
            raise NotImplementedError

        ori_target = ImageHelper.tonp(ss_dense)
        ori_target[ori_target == 255] = -1

        if self.aug_transform is not None:
            img, labelmap, gtmap = self.aug_transform(img, labelmap=labelmap, gtmap=ori_target)
        border_size = ImageHelper.get_size(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            labelmap = self.label_transform(labelmap)
            gtmap = self.label_transform(gtmap)

        meta = dict(
            ori_img_size=img_size,
            border_size=border_size,
            ori_target=ori_target
        )

        return dict(
            img=DataContainer(img, stack=self.is_stack),
            labelmap=DataContainer(labelmap, stack=self.is_stack),
            gtmap=DataContainer(gtmap, stack=self.is_stack),
            meta=DataContainer(meta, stack=False, cpu_only=True),
            name=DataContainer(name, stack=False, cpu_only=True),
        )

    def rasterize_scribbles(self, data, width, height):
        img = Image.new("L", (width, height), color=self.semseg_ignore_class)
        draw = ImageDraw.Draw(img)
        polylines = data
        for clsid, joints in polylines:
            if len(joints) > 1:
                draw.line(joints, clsid, self.stroke_width, joint="curve")
            for i in range(len(joints)):
                draw.ellipse((
                    joints[i][0] - self.stroke_width / 2, joints[i][1] - self.stroke_width / 2,
                    joints[i][0] + self.stroke_width / 2, joints[i][1] + self.stroke_width / 2),
                    clsid
                )
        return img

    def rasterize_clicks(self, data, width, height):
        img = Image.new("L", (width, height), color=self.semseg_ignore_class)
        draw = ImageDraw.Draw(img)
        for clsid, click in data:
            click = click[0]
            if self.stroke_width > 1:
                draw.ellipse((
                    click[0] - self.stroke_width / 2, click[1] - self.stroke_width / 2,
                    click[0] + self.stroke_width / 2, click[1] + self.stroke_width / 2),
                    clsid
                )
            else:
                draw.point(click, clsid)
        return img

    def name_from_index(self, index):
        return self.sample_names[index]

    def _parse_scribble(self, name, known_width=None, known_height=None):
        with zipfile.ZipFile(self.ds_scribbles_path, 'r') as f:
            data = f.read(name + '.xml')
        sample_xml = ElementTree.fromstring(data)
        assert sample_xml.tag == 'annotation', f'XML error in sample {name}'
        found_size = False
        polylines = []
        for i in range(len(sample_xml)):
            if sample_xml[i].tag == 'size':
                found_size = True
                found_width, found_height = False, False
                sample_xml_size = sample_xml[i]
                for j in range(len(sample_xml_size)):
                    if sample_xml_size[j].tag == 'width':
                        assert known_width is None or int(sample_xml_size[j].text) == known_width, \
                            f'XML error in sample {name}'
                        found_width = True
                    elif sample_xml_size[j].tag == 'height':
                        assert known_height is None or int(sample_xml_size[j].text) == known_height, \
                            f'XML error in sample {name}'
                        found_height = True
                assert found_width and found_height, f'XML error in sample {name}'
            if sample_xml[i].tag == 'polygon':
                polygon = sample_xml[i]
                polygon_class, polygon_points = None, []
                for j in range(len(polygon)):
                    polygon_entry = polygon[j]
                    if polygon_entry.tag == 'tag':
                        polygon_class = polygon_entry.text
                    elif polygon_entry.tag == 'point':
                        assert polygon_entry[0].tag == 'X' and polygon_entry[1].tag == 'Y', \
                            f'XML error in sample {name}'
                        polygon_points.append((int(polygon_entry[0].text), int(polygon_entry[1].text)))
                assert polygon_class is not None and len(polygon_points) > 0, f'XML error in sample {name}'
                polylines.append((self.cls_name_to_id[polygon_class], polygon_points))
        assert found_size and len(polylines) > 0, f'XML error in sample {name}'
        # coordinate tuples have (x,y) order
        return polylines

    @staticmethod
    def _sample_name(path):
        return path.split('/')[-1].split('.')[0]

    @property
    def num_classes(self):
        return 21

    @property
    def semseg_class_colors(self):
        return [
            (0, 0, 0),  # 'background'
            (128, 0, 0),  # 'plane'
            (0, 128, 0),  # 'bike'
            (128, 128, 0),  # 'bird'
            (0, 0, 128),  # 'boat'
            (128, 0, 128),  # 'bottle'
            (0, 128, 128),  # 'bus'
            (128, 128, 128),  # 'car'
            (64, 0, 0),  # 'cat'
            (192, 0, 0),  # 'chair'
            (64, 128, 0),  # 'cow'
            (192, 128, 0),  # 'table'
            (64, 0, 128),  # 'dog'
            (192, 0, 128),  # 'horse'
            (64, 128, 128),  # 'motorbike'
            (192, 128, 128),  # 'person'
            (0, 64, 0),  # 'plant'
            (128, 64, 0),  # 'sheep'
            (0, 192, 0),  # 'sofa'
            (128, 192, 0),  # 'train'
            (0, 64, 128),  # 'monitor'
        ]

    @property
    def semseg_class_names(self):
        return [
            'background',
            'plane',
            'bike',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'table',
            'dog',
            'horse',
            'motorbike',
            'person',
            'plant',
            'sheep',
            'sofa',
            'train',
            'monitor',
        ]

    @property
    def semseg_class_histogram(self):
        return self._semseg_class_histogram

    def _compute_histogram(self):
        clicks_histogram = [0, ] * self.num_classes
        for name in self.sample_names:
            for ds in (self.ds_clicks_fg, self.ds_clicks_bg):
                clicks = ds[name]
                for click in clicks:
                    clsid = click['cls']
                    clicks_histogram[clsid] += 1
        return clicks_histogram

    def _reduce_zero_label(self, labelmap):
        if not self.configer.get('data', 'reduce_zero_label'):
            return labelmap

        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1
        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

    def _encode_label(self, labelmap):
        labelmap = np.array(labelmap)

        shape = labelmap.shape
        encoded_labelmap = np.ones(shape=(shape[0], shape[1]), dtype=np.float32) * 255
        for i in range(len(self.configer.get('data', 'label_list'))):
            class_id = self.configer.get('data', 'label_list')[i]
            encoded_labelmap[labelmap == class_id] = i

        if self.configer.get('data', 'image_tool') == 'pil':
            encoded_labelmap = ImageHelper.np2img(encoded_labelmap.astype(np.uint8))

        return encoded_labelmap

if __name__ == "__main__":
    pass
