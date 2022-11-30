import torch
import os.path as osp
from torch.utils.data import Dataset
import os
from .piplines import Compose
import cv2


class SODDataset(Dataset):
    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir=None,
                 depth_dir=None,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 depth_map_suffix='.png',
                 data_root=None,
                 ):
        self.pipeline = Compose(pipeline)

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.depth_dir = depth_dir
        self.depth_map_suffix = depth_map_suffix
        self.data_root = data_root

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.depth_dir is None or osp.isabs(self.depth_dir)):
                self.depth_dir = osp.join(self.data_root, self.depth_dir)

        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir, self.seg_map_suffix,
                                               self.depth_dir, self.depth_map_suffix)


    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         depth_dir, depth_map_suffix):
        img_infos = []
        imgs_list = os.listdir(img_dir)
        for img in imgs_list:
            img_info = dict(filename=img)
            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
            if depth_dir is not None:
                depth_map = img.replace(img_suffix, depth_map_suffix)
                img_info['depth'] = dict(depth_map=depth_map)

            img_infos.append(img_info)
        return img_infos

    def get_ann_info(self, idx):

        return self.img_infos[idx]['ann']

    def get_depth_info(self, idx):

        return self.img_infos[idx]['depth']

    def pre_pipeline(self, results):

        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        results['depth_prefix'] = self.depth_dir

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        depth_info = self.get_depth_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, depth_info=depth_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __getitem__(self, idx):

        return self.prepare_img(idx)

    # for group sampler based on similar aspect ratio
    @property
    def aspect_ratio(self):
        aspect_ratio_list = []
        for info in self.img_infos:
            img = cv2.imread(osp.join(self.img_dir, info['filename']))
            h, w, _ = img.shape
            del img
            aspect_ratio_list.append(w/h)
        return aspect_ratio_list


    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        for idx in range(len(batch)-1):
            batch[idx] = torch.stack(batch[idx], dim=0)
        return batch