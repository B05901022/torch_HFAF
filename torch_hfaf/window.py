# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:14:49 2022

@author: AustinHsu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from qtorch.quant import fixed_point_quantize

DEFAULT_FXP_CONFIG = {
    # fxp_type: [word length, fraction length]
    "INPUT": [8,7],
    "WINDOW_COEFF": [8,6],
    "WINDOW_OUT": [8,7],
}

def get_window_func(win_func: str = "hann"):
    if win_func == "hann":
        return torch.hann_window
    elif win_func == "hamming":
        return torch.hamming_window
    elif win_func == "blackman":
        return torch.blackman_window
    else:
        raise NotImplementedError(f"Given window function {win_func} is invalid. Please choose from hann/hamming/blackman.")

class Window(nn.Module):

    def __init__(
        self,
        window_func: str = "hann",
        window_length: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
        post_window_pad_mode: str = 'constant',
        fxp_config: dict = DEFAULT_FXP_CONFIG,
        bypass_quant: bool = False,
        rounding_mode: str = "floor",
        ):
        super().__init__()
        self.window_func = get_window_func(window_func)
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fxp_config = fxp_config
        self.bypass_quant = bypass_quant
        self.rounding_mode = rounding_mode

        pre_window_pad_length = int( window_length // 2 )
        self.window_config = { # only supports zero_padding
            'kernel_size': (1, window_length),
            'stride': hop_length,
            'padding': (0, pre_window_pad_length),
        }

        post_window_pad_length = int( (n_fft - window_length) // 2 )
        self.post_window_pad = {
            'pad': (post_window_pad_length, post_window_pad_length),
            'mode': post_window_pad_mode,
            'value': 0 if post_window_pad_mode == 'constant' else None,
        }

        self.window_coeff = nn.Parameter(self.gen_window(), requires_grad=False)

    def forward(self, x):
        # x.shape = (batchsize=1, num_sample_points)
        #assert x.shape[0] == 1, f"Only support batchsize=1, {x.shape[0]} is given."
        out = self.fxp_quant(x, "INPUT")
        out = out.unsqueeze(1).unsqueeze(1) # (1, 1, 1, num_sample_points); F.unfold only supports 4-D tensor
        out = F.unfold(out, **self.window_config) # (1, window_length, num_frames)
        out = out * self.window_coeff # (1, window_length, num_frames)
        out = self.fxp_quant(out, "WINDOW_OUT")
        out = out.transpose(1,2) # (1, num_frames, window_length)
        out = F.pad(out, **self.post_window_pad) # (1, num_frames, n_fft)
        return out

    def fxp_quant(self, input, fxp_type=None, nearest_round=False):
        if not self.bypass_quant:
            if fxp_type is not None:
                wl, fl = self.fxp_config[fxp_type]
                rounding_mode = "nearest" if nearest_round else self.rounding_mode
                return fixed_point_quantize(input, wl, fl, rounding=rounding_mode)
            else:
                return input
        else:
            return input

    def gen_window(self):
        window_coeff = self.window_func(self.window_length)
        window_coeff = self.fxp_quant(window_coeff, "WINDOW_COEFF", True)
        window_coeff = window_coeff.unsqueeze(0).unsqueeze(-1)
        return window_coeff