#from torch import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from collections.abc import Iterator
from torch.utils.data.sampler import RandomSampler 

# class WeightedDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size, shuffle, weight):
#         super(WeightedDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)
#         self.weight = weight

#     def __call__(self, weight):
#         self.weight = weight
#         return self

#     def __iter__(self):
#         sampler = [WeightedRandomSampler(weight, num_sample=self.batch_size, replacement=True) 
#                 for weight in self.weight.T]
#         return iter(DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler))
    


class WeightedRandomSampler2(WeightedRandomSampler):
    def __iter__(self) -> Iterator[int]:
        effectiveNumSample = min(int(self.weights.sum()), self.num_samples)
        numSampleDelta = self.num_samples - effectiveNumSample
        rand_tensor_uw = []
        rand_tensor_w = []
        if effectiveNumSample > 0:
            rand_tensor_w = torch.multinomial(
                self.weights, effectiveNumSample, self.replacement, generator=self.generator)
            rand_tensor_w = rand_tensor_w.tolist()
        if numSampleDelta > 0:
            rand_tensor_uw = torch.randint(0, len(self.weights), (numSampleDelta,), generator=self.generator)
            rand_tensor_uw = rand_tensor_uw.tolist()
        yield from iter(rand_tensor_w + rand_tensor_uw)
        