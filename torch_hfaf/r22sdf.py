# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 19:57:36 2022

@author: AustinHsu
"""

import numpy as np
import torch
import torch.nn as nn
from qtorch.quant import fixed_point_quantize

DEFAULT_FXP_CONFIG = {
    "WN_COEFF": [6,4],
    "STAGE1_FFTOUT": [8,6],
    "STAGE2_FFTOUT": [8,6],
    "STAGE3_FFTOUT": [8,5],
    "STAGE4_FFTOUT": [8,5],
    "STAGE5_FFTOUT": [8,3],
    "STAGE6_FFTOUT": [8,3],
    "STAGE7_FFTOUT": [8,2],
    "STAGE8_FFTOUT": [8,2],
    "EXTRA_STAGE_COEFF": [4,2],
    "FINAL_OUT": [8,2],
    }

class R22SDF(nn.Module):
    
    def __init__(
        self, 
        n_fft: int = 512,
        evenodd_unpacking: bool = True,
        rfft: bool = True,
        pre_gain: int = 16,
        fxp_config: dict = DEFAULT_FXP_CONFIG,
        bypass_quant: bool = False,
        rounding_mode: str = "floor",
        ):
        super().__init__()
        self.n_fft = int(n_fft//2) if evenodd_unpacking else n_fft
        self.evenodd_unpacking = evenodd_unpacking
        self.rfft = rfft
        self.pre_gain = pre_gain # Gaining the input and de-gaining when output FFT
        self.fxp_config = fxp_config
        self.bypass_quant = bypass_quant
        self.rounding_mode = rounding_mode
        
        self.twiddle_factors = nn.Parameter(self.gen_twiddle(), requires_grad=False)
        self.extra_stage = nn.Parameter(self.gen_extrastage(), requires_grad=False)
        
    def forward(self, x):
        # x.shape = (batch, num_frames, n_fft)
        x = x*self.pre_gain
        if self.evenodd_unpacking:
            indices = torch.arange(self.n_fft, device=x.device)*2
            x_even = torch.index_select(x, -1, indices)
            x_odd = torch.index_select(x, -1, indices + 1)
            x = x_even + 1j*x_odd
        num_stages = np.log2(self.n_fft).astype(np.int32)
        for stage in range(1,num_stages+1):
            x = self.butterfly(x, stage)
            if stage != num_stages:
                x = self.mult(x, stage)
        # Bit reverse reordering
        x = self.reorder(x) # (batch, num_frames, n_fft)
        if self.evenodd_unpacking:
            x_even = 0.5 * (x+x.roll(-1,-1).flip(-1).conj()) # (batch, num_frames, n_fft/2)
            x_odd = -0.5j * (x-x.roll(-1,-1).flip(-1).conj()) # (batch, num_frames, n_fft/2)
            if not self.rfft:
                x_even = torch.cat((x_even,x_even),dim=-1)
                x_odd = torch.cat((x_odd,x_odd),dim=-1)
            else:
                x_even = torch.cat((x_even,x_even[:,:,:1]),dim=-1)
                x_odd = torch.cat((x_odd,x_odd[:,:,:1]),dim=-1)
            x = x_even + self.extra_stage * x_odd
        #x = x.unsqueeze(0)
        x_real = self.fxp_quant(x.real, "FINAL_OUT")
        x_imag = self.fxp_quant(x.imag, "FINAL_OUT")
        x = torch.complex(x_real, x_imag)
        x = x/self.pre_gain
        return x
    
    def fxp_quant(self, input, fxp_type=None, nearest_round=False):
        if not self.bypass_quant:
            if fxp_type is not None:
                wl, fl = self.fxp_config[fxp_type]
                rounding_mode = "nearest" if nearest_round else self.rounding_mode
                return fixed_point_quantize(input, wl, fl,rounding=rounding_mode)
            else:
                return input
        else:
            return input
    
    def w_N(self, nk, N):
        w_N_cos = torch.cos(torch.Tensor([-2*torch.pi*nk/N]))
        w_N_sin = torch.sin(torch.Tensor([-2*torch.pi*nk/N]))
        
        # Avoid overflowing problem (ones are more accurate)
        w_N_cos_fxp = self.fxp_quant(w_N_cos, "WN_COEFF", True)
        w_N_sin_fxp = self.fxp_quant(w_N_sin, "WN_COEFF", True)
        if w_N_cos_fxp == 1:
            return 1
        elif w_N_cos_fxp == -1:
            return -1
        elif w_N_sin_fxp == 1:
            w_N_cos_fxp, w_N_sin_fxp = torch.Tensor([ 0]), torch.Tensor([ 1])
        elif w_N_sin_fxp == -1:
            w_N_cos_fxp, w_N_sin_fxp = torch.Tensor([ 0]), torch.Tensor([-1])
        return complex(w_N_cos_fxp, w_N_sin_fxp) # complex == np.complex
    
    def gen_twiddle(self):
        assert self.n_fft % 4 == 0 or (self.n_fft/2) % 4 == 0, f"Given nFFT must be 4**k or 2*4**k, {self.n_fft} is invalid."
        
        num_stages = np.ceil(np.log(self.n_fft)/np.log(4)).astype(np.int32)
        n3_range = int(self.n_fft/4) # n = N/2 n1 + N/4 n2 + n3
        stage_repeat = 1 # how many times does the twiddle factors repeat
        stage_N = self.n_fft # the N in w_{N}^{nk}
        twiddle_factors = []
        for stage in range(num_stages-1):
            stage_x_twiddle_factor = [1 for _ in range(n3_range)]
            stage_x_twiddle_factor += \
                [self.w_N(2*i, stage_N) for i in range(n3_range)]
            stage_x_twiddle_factor += \
                [self.w_N(i, stage_N) for i in range(n3_range)]
            stage_x_twiddle_factor += \
                [self.w_N(3*i, stage_N) for i in range(n3_range)]
            stage_x_twiddle_factor = stage_x_twiddle_factor*stage_repeat
            
            twiddle_factors.append(stage_x_twiddle_factor)
            
            stage_repeat *= 4
            n3_range = int(n3_range/4)
            stage_N /= 4
        twiddle_factors = np.array(twiddle_factors, dtype=np.complex64)
        twiddle_factors = torch.from_numpy(twiddle_factors)
        
        return twiddle_factors
    
    def gen_extrastage(self):
        if self.evenodd_unpacking:
            if self.rfft:
                extra_stage = np.arange(self.n_fft+1)
            else:
                extra_stage = np.arange(self.n_fft*2)
            extra_stage = np.exp(-1j*np.pi/self.n_fft*extra_stage)
            extra_stage = torch.from_numpy(extra_stage)
            extra_stage_real = extra_stage.real.float()
            extra_stage_imag = extra_stage.imag.float()
            extra_stage_real = self.fxp_quant(extra_stage_real, "EXTRA_STAGE_COEFF", True)
            extra_stage_imag = self.fxp_quant(extra_stage_imag, "EXTRA_STAGE_COEFF", True)
            extra_stage = torch.complex(extra_stage_real, extra_stage_imag)
            extra_stage = extra_stage.unsqueeze(0)
        else:
            extra_stage = torch.Tensor([])
        return extra_stage
    
    def check_angle(self, twiddle_factors):
        nk = -twiddle_factors.angle()/2/torch.pi*self.n_fft
        nk[nk<0] += self.n_fft
        nk[nk==0] = 0
        return nk
    
    def butterfly(self, x, bf_stage):
        assert int(self.n_fft/2**(bf_stage)) > 0, f"Given stage {bf_stage} is invalid. Should be <= {np.log2(self.n_fft).astype(np.int32)} (nFFT={self.n_fft})."
        
        batchsize = x.shape[0]
        num_frames = x.shape[1]
        pairs = int(self.n_fft/2)
        view_shape = [batchsize, -1] + [2]*bf_stage + [int(self.n_fft/2**(bf_stage))]
        x = x.view(view_shape).transpose(-1,-2).reshape(batchsize,num_frames,pairs,2)
        add_branch = x.sum(dim=-1)
        sub_branch = x[:,:,:,0] - x[:,:,:,1]
        
        cat_shape = (batchsize, num_frames, 2**(bf_stage-1), int(self.n_fft/2**(bf_stage)))
        add_branch = add_branch.view(cat_shape)
        sub_branch = sub_branch.view(cat_shape)
        return torch.cat((add_branch,sub_branch),dim=-1).reshape(batchsize,num_frames,self.n_fft)
        
    
    def mult(self, x, stage):
        if stage % 2 == 1:
            # Trivial multiplication
            stage_repeat = 4**int(stage/2)
            inner_repeat = int(self.n_fft/4/stage_repeat)
            trivial_mult = ([1]*3*inner_repeat + [-1j]*inner_repeat )*stage_repeat
            trivial_mult = np.array([trivial_mult], dtype=np.complex64)
            trivial_mult = torch.from_numpy(trivial_mult).to(x.device)
            product = x * trivial_mult
        else:
            # Twiddle muliplication
            twiddle_stage = int(stage/2) - 1
            twiddle_mult = self.twiddle_factors[twiddle_stage]
            twiddle_mult = twiddle_mult.unsqueeze(0)#.to(x.device)
            product = x * twiddle_mult
        product_real = self.fxp_quant(product.real, f"STAGE{stage}_FFTOUT")
        product_imag = self.fxp_quant(product.imag, f"STAGE{stage}_FFTOUT")
        product = torch.complex(product_real, product_imag)
        return product
    
    def reorder(self, x):
        indices = np.array([int(np.binary_repr(i, width=int(np.log2(self.n_fft)))[::-1],2) for i in range(self.n_fft)])
        indices = torch.from_numpy(indices).long()
        return x[:,:,indices]