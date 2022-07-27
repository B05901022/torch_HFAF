# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:14:49 2022

@author: AustinHsu
"""

import torch.nn as nn
from qtorch.quant import fixed_point_quantize

DEFAULT_FXP_CONFIG = {
	'POWER_OUT': [8,7],
	'SPECTRUM_OUT': [8,7],
}

class PowerSpectrum(nn.Module):

	def __init__(
		self,
		fxp_config: dict = DEFAULT_FXP_CONFIG,
		bypass_quant: bool = False,
        rounding_mode: str = "floor",
		):
		super().__init__()
		self.fxp_config = fxp_config
		self.bypass_quant = bypass_quant
		self.rounding_mode = rounding_mode

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
		x_imag_pow = x.imag ** 2
		x_real_pow = x.real ** 2
		x_imag_pow = self.fxp_quant(x_imag_pow, 'POWER_OUT')
		x_real_pow = self.fxp_quant(x_real_pow, 'POWER_OUT')
		out = x_imag_pow + x_real_pow
		out = self.fxp_quant(out, 'SPECTRUM_OUT')
		return out