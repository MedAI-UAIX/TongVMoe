import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from architecture.abstract_arch import AbsArchitecture


class MMoE(AbsArchitecture):


    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

        self.img_size = self.kwargs['img_size']
        self.input_size = np.array(self.img_size, dtype=int).prod()
        self.num_experts = self.kwargs['num_experts'][0]
        self.experts_shared = encoder_class()
        # self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, self.num_experts),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})

    def forward(self, inputs, task=None):
        feature = self.experts_shared(inputs)
        experts_shared_rep = torch.stack([feature for _ in range(self.num_experts)])
        out = {}

        selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1))
        gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
        gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
        out[task] = self.decoders[task](gate_rep)
        return out



    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad(set_to_none=False)



