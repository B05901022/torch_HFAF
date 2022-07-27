# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:29:00 2022

@author: AustinHsu
"""

from tqdm import tqdm

from transform import AudioPreprocessing, GTAudioProcessing

def fit_config(
    test_dataloader,
    config,
    ):
    
    for b_idx, data in enumerate(tqdm(test_dataloader)):
        
    
    return