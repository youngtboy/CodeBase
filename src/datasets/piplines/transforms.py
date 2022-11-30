# modify from open-mmlab
from numpy import random
from src.datasets.piplines.utils import *
import copy


class Resize(object):

    def __init__(self,
                 img_scale=None,
                 ratio_range=None,
                 keep_ratio=False,
                 multiscale_mode='range'):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.multiscale_mode = multiscale_mode

    @staticmethod
    def random_select(img_scales):

        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):

        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
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

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = imrescale(results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = imresize(results['img'], results['scale'], return_scale=True)

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):

        if self.keep_ratio:
            gt_seg = imrescale(results['gt_semantic_seg'], results['scale'], interpolation='nearest')
        else:
            gt_seg = imresize(results['gt_semantic_seg'], results['scale'],  interpolation='nearest')
        results['gt_semantic_seg'] = gt_seg

    def _resize_depth(self, results):

        if self.keep_ratio:
            depth_map = imrescale(results["depth_map"], results['scale'], interpolation='nearest')
        else:
            depth_map = imresize(results["depth_map"], results['scale'],  interpolation='nearest')
        results["depth_map"] = depth_map

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        self._resize_depth(results)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(img_scale={self.img_scale})'


class RandomFlip(object):

    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

        if self.direction is 'horizontal':
            self.flipcode = 1
        elif self.direction is 'hvertical':
            self.flipcode = 0

    def __call__(self, results):

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip

        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction

        if results['flip']:
            # flip image
            results['img'] = cv2.flip(results['img'], flipCode=self.flipcode)

            # flip segs
            results['gt_semantic_seg'] = cv2.flip(results['gt_semantic_seg'], flipCode=self.flipcode).copy()

            # flip depths
            results["depth_map"] = cv2.flip(results["depth_map"], flipCode=self.flipcode).copy()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob}, direction={self.direction})'


class Pad(object):
    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, depth=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        pad_val = self.pad_val.get('img', 0)

        if self.pad_to_square:
            max_size = max(results["img"].shape[:2])
            self.size = (max_size, max_size)
        if self.size is not None:
            padded_img = impad(
                results["img"], shape=self.size, pad_val=pad_val)
        elif self.size_divisor is not None:
            padded_img = impad_to_multiple(
                results["img"], self.size_divisor, pad_val=pad_val)

        results["img"] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):

        pad_val = self.pad_val.get('masks', 0)
        results['gt_semantic_seg'] = impad(results['gt_semantic_seg'], shape=results['pad_shape'][:2], pad_val=pad_val)

    def _pad_depth(self, results):

        pad_val = self.pad_val.get('seg', 0)
        results["depth_map"] = impad(results["depth_map"], shape=results['pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):

        self._pad_img(results)
        self._pad_seg(results)
        self._pad_seg(results)
        return results


class Normalize(object):

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):

        results['img'] = imnormalize(results['img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class RandomCrop(object):

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):

        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        results['gt_semantic_seg'] = self.crop(results["gt_semantic_seg"], crop_bbox)

        return results


class ColorJitter(object):
    """
        Apply photometric distortion to image sequentially, every transformation
        is applied with a probability of 0.5. The position of random contrast is in
        second or second to last.

        1. random brightness
        2. random contrast (mode 0)
        3. convert color from BGR to HSV
        4. random saturation
        5. random hue
        6. convert color from HSV to BGR
        7. random contrast (mode 1)
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower, self.saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(self, img):
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = hsv2bgr(img)
        return img

    def __call__(self, results):

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results