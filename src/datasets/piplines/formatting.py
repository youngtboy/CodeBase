import torch
import numpy as np


def to_tensor(data):

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):

        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results


class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


class Collect(object):

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
            if key in results.keys():
                img_meta[key] = results[key]
        data['img_metas'] = img_meta
        result = []
        for key in self.keys:
            data[key] = results[key]

            result.append(results[key])

        result.append(data['img_metas'])
        return tuple(result)

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
