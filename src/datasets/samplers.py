from torch.utils.data import Sampler
import numpy as np
import torch
import torch.distributed as dist
import math


__all__ = [
    "GroupSampler",
    "DistributedGroupSampler",
    "DistributedSampler",
    "BatchSampler",
]


# Mostly copy from openmmlab
# https://github.com/open-mmlab/mmdetection/blob/e71b499608e9c3ccd4211e7c815fa20eeedf18a2/mmdet/datasets/samplers/group_sampler.py#L10
class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'aspect_ratio_flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.aspect_ratio_flag.astype(np.int64)  
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for _, size in enumerate(self.group_sizes): 
            self.num_samples += int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes): 
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0] 
            assert len(indice) == size
            np.random.shuffle(indice)  
            num_extra = int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, np.random.choice(indice, num_extra)])  
            indices.append(indice)
        indices = np.concatenate(indices)  
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ] 
        indices = np.concatenate(indices)  
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        assert hasattr(self.dataset, 'aspect_ratio_flag')
        self.flag = self.dataset.aspect_ratio_flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, _ in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                indice = indice[list(torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(torch.randperm(len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# modify from Deformable-DETR and Pytorch
# https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/datasets/samplers.py#L16
# https://github.com/pytorch/pytorch/blob/419ef2cdcfe84442de5232739284c6a51a18632f/torch/utils/data/distributed.py#L13
class DistributedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank = None, shuffle = True, seed = 0, drop_last=False):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist() 
        else:
            indices = list(range(len(self.dataset))) 

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample 间隔采样
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # 连续采样
        # offset = self.num_samples * self.rank
        # indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# Mostly copy from Pytorch
# https://github.com/pytorch/pytorch/blob/caf3d5319f15e47363fe36856326f5e4ab3303e1/torch/utils/data/sampler.py#L210
class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or  batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size  
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size 
