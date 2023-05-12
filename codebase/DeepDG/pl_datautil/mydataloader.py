# coding=utf-8
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from typing import Iterator, List, Optional, Union
from collections import Counter
from operator import itemgetter
import numpy as np

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class DistributedWeightedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     dist.init_process_group()
        #     num_replicas = dist.get_world_size()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError("Requires distributed package to be available")
        #     rank = dist.get_rank()
        # self.dataset = dataset
        # self.num_replicas = num_replicas
        # self.rank = rank
        # self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        ic(self.num_samples, self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = True
        self.g = torch.Generator()
        self.g.manual_seed(self.epoch)
        # self.shuffle = shuffle


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = float(len(targets)) / class_sample_count.double()
        ic(class_sample_count, weight)
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        ic('---------------iter---------------')
        ic(self.epoch, self.seed, self.total_size)
        self.g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        ic(len(indices))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        ic(len(indices))
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.labels
        targets = targets[indices]
        assert len(targets) == self.num_samples
        ic(type(targets))
        weights = self.calculate_weights(targets)
        ic(weights)

        subsampled_balanced_indices = torch.multinomial(weights, self.total_size, self.replacement).tolist()
        ic(len(subsampled_balanced_indices))
        subsampled_balanced_indices = [indices[idx] for idx in range(len(subsampled_balanced_indices))]
        ic(len(subsampled_balanced_indices))

        return iter(subsampled_balanced_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        # self.epoch = epoch

class DynamicBalanceClassSampler(DistributedSampler):
    """
    This kind of sampler can be used for classification tasks with significant
    class imbalance.
    The idea of this sampler that we start with the original class distribution
    and gradually move to uniform class distribution like with downsampling.
    Let's define :math: D_i = #C_i/ #C_min where :math: #C_i is a size of class
    i and :math: #C_min is a size of the rarest class, so :math: D_i define
    class distribution. Also define :math: g(n_epoch) is a exponential
    scheduler. On each epoch current :math: D_i  calculated as
    :math: current D_i  = D_i ^ g(n_epoch),
    after this data samples according this distribution.
    Notes:
         In the end of the training, epochs will contain only
         min_size_class * n_classes examples. So, possible it will not
         necessary to do validation on each epoch. For this reason use
         ControlFlowCallback.
    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from catalyst.data import DynamicBalanceClassSampler
        >>> from torch.utils import data
        >>> features = torch.Tensor(np.random.random((200, 100)))
        >>> labels = np.random.randint(0, 4, size=(200,))
        >>> sampler = DynamicBalanceClassSampler(labels)
        >>> labels = torch.LongTensor(labels)
        >>> dataset = data.TensorDataset(features, labels)
        >>> loader = data.dataloader.DataLoader(dataset, batch_size=8)
        >>> for batch in loader:
        >>>     b_features, b_labels = batch
    Sampler was inspired by https://arxiv.org/abs/1901.06783
    """

    def __init__(
        self,
        labels: List[Union[int, str]],
        exp_lambda: float = 0.9,
        start_epoch: int = 0,
        max_d: Optional[int] = None,
        mode: Union[str, int] = "downsampling",
        ignore_warning: bool = False,
    ):
        """
        Args:
            labels: list of labels for each elem in the dataset
            exp_lambda: exponent figure for schedule
            start_epoch: start epoch number, can be useful for multi-stage
            experiments
            max_d: if not None, limit on the difference between the most
            frequent and the rarest classes, heuristic
            mode: number of samples per class in the end of training. Must be
            "downsampling" or number. Before change it, make sure that you
            understand how does it work
            ignore_warning: ignore warning about min class size
        """
        assert isinstance(start_epoch, int)
        super().__init__(labels)
        self.exp_lambda = exp_lambda
        if max_d is None:
            max_d = np.inf
        self.max_d = max_d
        self.epoch = start_epoch
        labels = np.array(labels)
        samples_per_class = Counter(labels)
        self.min_class_size = min(samples_per_class.values())

        if self.min_class_size < 100 and not ignore_warning:
            LOGGER.warning(
                f"the smallest class contains only"
                f" {self.min_class_size} examples. At the end of"
                f" training, epochs will contain only"
                f" {self.min_class_size * len(samples_per_class)}"
                f" examples"
            )

        self.original_d = {
            key: value / self.min_class_size for key, value in samples_per_class.items()
        }
        self.label2idxes = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, int):
            self.min_class_size = mode
        else:
            assert mode == "downsampling"

        self.labels = labels
        current_d = {
            key: min(value ** self._exp_scheduler(), self.max_d)
            for key, value in self.original_d.items()
        }
        samples_per_classes = {
            key: int(value * self.min_class_size) for key, value in current_d.items()
        }
        self.samples_per_classes = samples_per_classes
        self.length = np.sum(list(samples_per_classes.values()))

        self._update()

    def _update(self) -> None:
        """Update d coefficients."""
        # current_d = {
        #     key: min(value ** self._exp_scheduler(), self.max_d)
        #     for key, value in self.original_d.items()
        # }
        # samples_per_classes = {
        #     key: int(value * self.min_class_size) for key, value in current_d.items()
        # }
        # self.samples_per_classes = samples_per_classes
        # ic(self.samples_per_classes)
        # self.length = np.sum(list(samples_per_classes.values()))
        # ic(self.length)
        self.epoch += 1

    def _exp_scheduler(self) -> float:
        return self.exp_lambda ** self.epoch

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            iterator of indices of stratified sample
        """
        ic('---------------iter-----------------')
        indices = []
        for key in sorted(self.label2idxes):
            ic(key)
            samples_per_class = self.samples_per_classes[key]
            replace_flag = samples_per_class > len(self.label2idxes[key])
            indices += np.random.choice(
                self.label2idxes[key], samples_per_class, replace=replace_flag
            ).tolist()
        ic(len(indices))
        assert len(indices) == self.length
        np.random.shuffle(indices)
        self._update()
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.dataset = dataset
        # self.n_steps = n_steps
        # ic(dataset.__len__())

        self._infinite_iterator = super().__iter__()
        
        weights = self.dataset.get_sample_weights()[0]
        # ic(weights)
        if weights == None:
            weights = torch.ones(len(self.dataset))
        
        sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                            replacement=True,
                                                            num_samples=self.batch_size)
        # else:
        #     sampler = torch.utils.data.RandomSampler(dataset,
        #                                              replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=self.batch_size,
            drop_last=True)
        self._infinite_iterator = iter(_InfiniteSampler(torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            batch_sampler=batch_sampler,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            # drop_last=True,
        )))
        # else:
        #     self._infinite_iterator = iter(torch.utils.data.DataLoader(
        #         dataset,
        #         batch_size=batch_size,
        #         num_workers=num_workers,
        #         pin_memory=True,
        #         persistent_workers=True
        #     ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)
        # return self

    def __next__(self):
        try:
            batch = next(self._infinite_iterator)
        except StopIteration:
            self._infinite_iterator = super().__iter__()
            batch = next(self._infinite_iterator)
        return batch
    
    # def __len__(self):
    #     return self.n_steps

