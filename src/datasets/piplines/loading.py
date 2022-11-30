import os.path as osp
import cv2
import numpy as np


class LoadImageFromFile(object):

    def __init__(self,
                 color_type=1,
                 ):

        self.color_type = color_type

    def __call__(self, results):

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = cv2.imread(filename, flags=self.color_type)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return self.__class__.__name__


class LoadAnnotations(object):

    def __init__(self,
                 reduce_zero_label=False):
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        gt_semantic_seg = cv2.imread(filename, flags=0)
        results['gt_semantic_seg'] = gt_semantic_seg/255  # convert to [0, 1]

        return results

    def __repr__(self):
        return self.__class__.__name__


class LoadDepthFromFile(object):

    def __init__(self,
                 color_type=0,
                 ):

        self.color_type = color_type

    def __call__(self, results):

        if results.get('depth_prefix') is not None:
            filename = osp.join(results['depth_prefix'], results['depth_info']['depth_map'])
        else:
            filename = results['depth_info']['depth_map']

        if osp.exists(filename):
            depth_map = cv2.imread(filename, flags=self.color_type)
        else:
            depth_map = cv2.imread(filename.replace('png', 'bmp'), flags=self.color_type)
        depth_map = depth_map/255  # convert to [0, 1]
        results['depth_map'] = depth_map

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(color_type={self.color_type})'