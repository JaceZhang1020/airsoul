import sys
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict
from .tools import log_debug, log_warn

class DistStatistics(object):
    """
    Provide distributed statistics over GPUs
    """
    def __init__(self, *keys):
        self.keys = keys
        self.reset()

    def reset(self):
        self._data = dict()
        self._count = dict()
        for key in self.keys:
            self._data[key] = []
            self._count[key] = []

    def gather(self, device, count=None, **kwargs):
        """
        Count being regarded as the number of samples behind each of the value
        if value is an array, then it is regarded as the number of samples behind each of the value
        """
        if(count is None):
            fcount = torch.Tensor([1]).to(device)
        elif(isinstance(count, list) or isinstance(count, tuple)):
            fcount = torch.Tensor(count).to(device)
        elif(isinstance(count, torch.Tensor)):
            fcount = count.clone().to(device)
        else:
            fcount = torch.Tensor([count]).to(device)
        
        for key, value in kwargs.items():
            if(key not in self._data):
                log_warn(f"Key {key} not registered in DistStatistics object")
            
            if isinstance(value, list) or isinstance(value, tuple):
                fvalue = torch.stack(value, dim=0).to(device)
            elif(isinstance(value, torch.Tensor)):
                fvalue = value.clone().to(device)
            else:
                fvalue = torch.Tensor(value).to(device)

            assert fcount.numel() == 1 or fcount.numel() == fvalue.numel(), \
                f"dimension mismatch between statistic count {fcount.shape} and value {fvalue.shape}"

            if torch.isinf(fvalue).any() or torch.isnan(fvalue).any():
                log_warn(f"'Device:{device}' stating '{key}' has inf/NaN")
                fvalue = torch.where(torch.isfinite(fvalue), 
                                     fvalue, torch.zeros_like(fvalue))
            
            # Make sure both has the same dimension
            fvalue = fvalue.squeeze()
            if(fcount.ndim > fvalue.ndim):
                fcount = fcount.squeeze()
            while(fcount.ndim < fvalue.ndim):
                fcount = fcount.unsqueeze(-1)
            
            #loss matrix dim is [2,T//downsample_length], first row is position_wise mean, second row is variance.
            gathered_tensors = [torch.zeros_like(fvalue) for _ in range(dist.get_world_size())]
            gathered_counts = [torch.zeros_like(fcount) for _ in range(dist.get_world_size())]

            # gather values from all devices
            dist.all_gather(gathered_tensors, fvalue.data)
            dist.all_gather(gathered_counts, fcount.data)

            #If device num is 8, self._data[key] has 8 elements, each element is a tensor with shape [2,T//downsample_length]
            #Each element can be a length-1, length-2 tensors
            self._data[key].extend(gathered_tensors)
            self._count[key].extend(gathered_counts)

    def _stat(self, key):
        # Check if the data needs padding
        need_padding = False
        length = self._data[key][0].numel()
        for i in range(1, len(self._data[key])):
            if(self._data[key][i].numel() != length):
                need_padding = True

        # Pad if the sequences are not uniform, otherwise stack them
        if(self._need_padding[key]):
            value = rnn_utils.pad_sequence(self._data[key], batch_first=True, padding_value=0)
            count = rnn_utils.pad_sequence(self._count[key], batch_first=True, padding_value=0)
        else:
            value = torch.stack(self._data[key], dim=0)
            counts = torch.stack(self._count[key], dim=0)

        sum_cnt = torch.clip(torch.sum(counts, dim=0), min=1.0e-6)
        x_mean = torch.sum(value * counts, dim=0, keepdim=False) / sum_cnt
        x2_mean = torch.sum(value ** 2 * counts, dim=0, keepdim=False) / sum_cnt

        var = torch.sqrt(x2_mean - x_mean ** 2)

        return x_mean, var, sum_cnt

    def __call__(self, reset=True):
        stat_res = dict()
        for key in self.keys:
            mean,std,cnt = self._stat(key)
            # 95% Confidence Bound For Mean
            bound = 2.0 * std / torch.sqrt(cnt)
            if(mean.numel() < 2):
                mean = mean.squeeze().item()
                std = std.squeeze().item()
                bound = bound.squeeze().item()
            else:
                mean = mean.squeeze().tolist()
                std = std.squeeze().tolist()
                bound = bound.squeeze().tolist()
            stat_res[key] = {"mean":mean,"std":std,'cnt':cnt,
                    'bound':bound}
        if(reset):
            self.reset()
        return stat_res
