# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:14:49 2022

@author: AustinHsu
"""

import torch.nn as nn
import torchaudio
from qtorch.quant import fixed_point_quantize

DEFAULT_FXP_CONFIG = {
    'MEL_COEFF': [8,6],
    'MEL_OUT': [10,6],
}

class MelFBank(nn.Module):

    def __init__(
        self,
        n_freqs: int = 257,
        n_mels: int = 40,
        sample_rate: int = 16000,
        mel_scale: str = 'htk',
        fxp_config: dict = DEFAULT_FXP_CONFIG,
        bypass_quant: bool = False,
        rounding_mode: str = "floor",
        ):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_scale = mel_scale
        self.fxp_config = fxp_config
        self.bypass_quant = bypass_quant
        self.rounding_mode = rounding_mode

        self.melscale_fbank = nn.Parameter(self.gen_mel(), requires_grad=False)

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

    def forward(self, x):
        # x.shape = (1, 257, num_frames)
        out = x @ self.melscale_fbank
        out = self.fxp_quant(out, 'MEL_OUT')
        out = out.transpose(1,2)
        return out

    def gen_mel(self):
        melscale_fbank = torchaudio.functional.melscale_fbanks(
            n_freqs = self.n_freqs,
            f_min = 0,
            f_max = self.sample_rate // 2,
            n_mels = self.n_mels,
            sample_rate = self.sample_rate,
            norm = None,
            mel_scale = self.mel_scale,
            )
        melscale_fbank = self.fxp_quant(melscale_fbank, 'MEL_COEFF', True)
        return melscale_fbank