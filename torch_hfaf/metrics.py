# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 23:01:57 2022

@author: AustinHsu
"""

import torch
import torchaudio.functional as F

def l2_dist(complex_a, complex_b):
    """Usage: FFT/MelSpectrogram"""
    mag_a = complex_a.abs() if complex_a.is_complex() else complex_a
    mag_b = complex_b.abs() if complex_b.is_complex() else complex_b
    return (mag_a-mag_b).pow(2).sum(dim=(1,2)).mean()

def ssim(complex_a, complex_b):
    """Usage: FFT/MelSpectrogram"""
    mag_a = complex_a.abs() if complex_a.is_complex() else complex_a
    mag_b = complex_b.abs() if complex_b.is_complex() else complex_b
    mu_x = mag_a.mean(dim=(1,2))
    mu_y = mag_b.mean(dim=(1,2))
    sigma_x_sqr = torch.pow(mag_a-mu_x.unsqueeze(-1).unsqueeze(-1),2).mean(dim=(1,2))
    sigma_y_sqr = torch.pow(mag_b-mu_y.unsqueeze(-1).unsqueeze(-1),2).mean(dim=(1,2))
    sigma_xy = ((mag_a-mu_x.unsqueeze(-1).unsqueeze(-1))*(mag_b-mu_y.unsqueeze(-1).unsqueeze(-1))).mean(dim=(1,2))
    const_L = mag_a.view(mag_a.shape[0],-1).max(-1).values
    #const_c1 = 1/torch.sqrt(const_L)
    #const_c2 = 1/torch.sqrt(const_L)
    ssim_values = (2*mu_x*mu_y+const_L)/(mu_x**2+mu_y**2+const_L)
    ssim_values *= (2*sigma_xy+const_L)/(sigma_x_sqr+sigma_y_sqr+const_L)
    return ssim_values.mean()

def psnr(complex_a, complex_b):
    """Usage: FFT/MelSpectrogram"""
    mag_a = complex_a.abs() if complex_a.is_complex() else complex_a
    mag_b = complex_b.abs() if complex_b.is_complex() else complex_b
    X_max = mag_a.view(mag_a.shape[0],-1).max(-1).values
    psnr_values = 10*torch.log10(X_max.unsqueeze(-1).unsqueeze(-1)**2/((mag_a-mag_b).pow(2).mean(dim=(1,2))))
    return psnr_values.mean()

def mse(complex_a, complex_b):
    """Usage: FFT/MelSpectrogram"""
    mag_a = complex_a.abs() if complex_a.is_complex() else complex_a
    mag_b = complex_b.abs() if complex_b.is_complex() else complex_b
    mse_values = (mag_a-mag_b).pow(2).mean(dim=(1,2))
    return mse_values.mean()

def nmse(complex_a, complex_b):
    """Usage: FFT/MelSpectrogram"""
    mag_a = complex_a.abs() if complex_a.is_complex() else complex_a
    mag_b = complex_b.abs() if complex_b.is_complex() else complex_b
    nmse_values = (mag_a-mag_b).pow(2).sum(dim=(1,2))
    nmse_values /= mag_a.pow(2).sum(dim=(1,2))
    return nmse_values.mean()

def cd(complex_a, complex_b):
    """Usage: FFT"""
    complex_a = complex_a.abs() + 1e-6
    complex_b = complex_b.abs() + 1e-6
    cep_a = torch.fft.irfft(complex_a.log(), n=512)
    cep_b = torch.fft.irfft(complex_b.log(), n=512)
    return (cep_a-cep_b).pow(2).sum(dim=(1,2)).sqrt().mean()

def mcd(complex_a, complex_b):
    """Usage: FFT/MelSpectrogram"""
    mel_fbank = F.melscale_fbanks(257, 0, 8000, 40, 16000).to(complex_a.device)
    dct = F.create_dct(40, 40, "ortho").to(complex_a.device)
    cep_a = complex_a.abs().pow(2) @ mel_fbank if complex_a.is_complex() else complex_a.transpose(1,2)
    cep_b = complex_b.abs().pow(2) @ mel_fbank if complex_b.is_complex() else complex_b.transpose(1,2)
    cep_a = (cep_a + 1e-6).log() @ dct
    cep_b = (cep_b + 1e-6).log() @ dct
    return (cep_a-cep_b).pow(2).sum(dim=(1,2)).sqrt().mean()

def intel(complex_a, complex_b):
    # Objective Intelligibility Measure
    """Usage: FFT/MelSpectrogram"""
    # complex_a: (batch, time or frame, freq)
    # complex_b: (batch, time or frame, freq)
    mag_a = complex_a.abs() if complex_a.is_complex() else complex_a
    mag_b = complex_b.abs() if complex_b.is_complex() else complex_b
    mu_a = mag_a.mean(dim=2).unsqueeze(-1)
    mu_b = mag_b.mean(dim=2).unsqueeze(-1)
    norm_a = mag_a - mu_a
    norm_b = mag_b - mu_b
    djm = torch.einsum('btf,bft->bt', norm_a, norm_b.transpose(1,2))
    djm /= (norm_a.norm(2,dim=2)*norm_b.norm(2,dim=2)+1e-6)
    return djm.mean()

def log_intel(complex_a, complex_b, eps=1e-6):
    # Objective Intelligibility Measure
    """Usage: LogMelSpectrogram"""
    # complex_a: (batch, time or frame, freq)
    # complex_b: (batch, time or frame, freq)
    mag_a = (complex_a + eps).log()
    mag_b = (complex_b + eps).log()
    mu_a = mag_a.mean(dim=2).unsqueeze(-1)
    mu_b = mag_b.mean(dim=2).unsqueeze(-1)
    norm_a = mag_a - mu_a
    norm_b = mag_b - mu_b
    djm = torch.einsum('btf,bft->bt', norm_a, norm_b.transpose(1,2))
    djm /= (norm_a.norm(2,dim=2)*norm_b.norm(2,dim=2)+1e-6)
    return djm.mean()