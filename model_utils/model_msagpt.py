import math
import copy
import torch
from torch.nn import functional as F
import torch.nn as nn

from .model_proteinglm_clm import ProteinGLMForGeneration


class MSAGPT(ProteinGLMForGeneration):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(
            args,
            transformer=transformer,
            **kwargs
        )

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('MSAGPT-inference', 'MSAGPT inference Configurations')
        return super().add_model_specific_args(parser)

class FineTuneMSAGPT(MSAGPT):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(
            args,
            transformer=transformer,
            **kwargs
        )
        pass