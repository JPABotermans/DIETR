"""
------------------------------------------------------------------------
DIETR
Copyright (c) 2026 Koen Botermans
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
------------------------------------------------------------------------
Modified from pytorch-image-models (https://github.com/huggingface/pytorch-image-models)
Copyright (c) 2020 Ross Wightman. 
Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
------------------------------------------------------------------------
"""

import torch
from copy import deepcopy
import math

class ModelEMA(object):
    def __init__(self, model: torch.nn.Module, decay: float=0.999, warmups: int=1000, start: int=0):
        super().__init__()

        self.module = deepcopy(model).eval()

        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates
        if warmups == 0:
            self.decay_fn = lambda x: decay
        else:
            self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)


    def update(self, model: torch.nn.Module):
        
        if self.before_start < self.start:
            self.before_start += 1
            return
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self, ):
        return self.module.state_dict()

    def load_state_dict(self, state, strict: bool = True):
        if not isinstance(state, dict):
            return
        self.module.load_state_dict(state, strict=strict)
        if 'updates' in state:
            self.updates = state['updates']

    def __call__(self, x_trn_batch: torch.Tensor, y_trn_batch: torch.Tensor | None = None):
        return self.module(x_trn_batch, y_trn_batch)

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'
    
    def eval(self):
        pass 
    
    def train(self):
        pass 