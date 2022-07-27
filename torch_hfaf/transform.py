# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:14:49 2022

@author: AustinHsu
"""

import torch.nn as nn
import yaml
import copy

from .window import Window
from .r22sdf import R22SDF
from .powerspec import PowerSpectrum
from .melfbank import MelFBank

def get_default_config():
    config = {
    "enabled_modules": ["window","fft","powerspec","mel"],
    "window": {
        "window_func": "hann",
        "window_length": 400,
        "hop_length": 160,
        "n_fft": 512,
        "post_window_pad_mode": "constant",
        "fxp_config": {
            "INPUT": [16,15],
            "WINDOW_COEFF": [16,14],
            "WINDOW_OUT": [16,15],
            },
        "bypass_quant": False,
        "rounding_mode": "floor",
        },
    "fft": {
        "n_fft": 512,
        "evenodd_unpacking": True,
        "rfft": True,
        "pre_gain": 32,
        "fxp_config": {
            "WN_COEFF": [16,14],
            "STAGE1_FFTOUT": [16,10],
            "STAGE2_FFTOUT": [16,10],
            "STAGE3_FFTOUT": [16,8],
            "STAGE4_FFTOUT": [16,8],
            "STAGE5_FFTOUT": [16,6],
            "STAGE6_FFTOUT": [16,6],
            "STAGE7_FFTOUT": [16,4],
            "STAGE8_FFTOUT": [16,4],
            "EXTRA_STAGE_COEFF": [16,14],
            "FINAL_OUT": [16,4],
            },
        "bypass_quant": False,
        "rounding_mode": "floor",
        },
    "powerspec": {
        "fxp_config": {
            "POWER_OUT": [31,20],
            "SPECTRUM_OUT": [32,20],
            },
        "bypass_quant": False,
        "rounding_mode": "floor",
        },
    "mel": {
        "n_freqs": 257,
        "n_mels": 40,
        "sample_rate": 16000,
        "mel_scale": "htk",
        "fxp_config": {
            "MEL_COEFF": [16,14],
            "MEL_OUT": [32,20],
            },
        "bypass_quant": False,
        "rounding_mode": "floor",
        },
    }
    return config

class AudioPreprocessing(nn.Module):
    
    def __init__(self, config=get_default_config()):
        super().__init__()
        self.config = copy.deepcopy(config)
        
        # --- Operations ---
        self.chained_ops = []
        if "window" in self.config['enabled_modules']:
            self.chained_ops.append(
                Window(**self.config['window'])
                )
        if "fft" in self.config['enabled_modules']:
            self.chained_ops.append(
                R22SDF(**self.config['fft'])
                )
        if "powerspec" in self.config['enabled_modules']:
            self.chained_ops.append(
                PowerSpectrum(**self.config['powerspec'])
                )
        if "mel" in self.config['enabled_modules']:
            self.chained_ops.append(
                MelFBank(**self.config['mel'])
                )
        self.chained_ops = nn.Sequential(*self.chained_ops)
            
    def forward(self, x):
        return self.chained_ops(x)
    
class GTAudioProcessing(nn.Module):
    "Ground Truth version of AudioProcessing (fixed-point bypassed)"
    def __init__(self, config=get_default_config()):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.bypass_fxpconfig()
        
        # --- Operations ---
        self.chained_ops = []
        if "window" in self.config['enabled_modules']:
            self.chained_ops.append(
                Window(**self.config['window'])
                )
        if "fft" in self.config['enabled_modules']:
            self.chained_ops.append(
                R22SDF(**self.config['fft'])
                )
        if "powerspec" in self.config['enabled_modules']:
            self.chained_ops.append(
                PowerSpectrum(**self.config['powerspec'])
                )
        if "mel" in self.config['enabled_modules']:
            self.chained_ops.append(
                MelFBank(**self.config['mel'])
                )
        self.chained_ops = nn.Sequential(*self.chained_ops)
            
    def forward(self, x):
        return self.chained_ops(x)
    
    def bypass_fxpconfig(self):
        for key, value in self.config.items():
            if key in ["window", "fft", "powerspec", "mel"]:
                self.config[key]["bypass_quant"] = True