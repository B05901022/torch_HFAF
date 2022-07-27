# torch_HFAF

A simulation of R2\superscript{2}SDF FFT on PyTorch with fixed-point analysis capability.

## Installation

### Install QPyTorch

Install [QPyTorch][QPyTorch-Ours] from the following fork to enable fixed-point floor rounding.

```bash
git clone https://github.com/B05901022/QPyTorch.git
cd QPyTorch
python setup.py install
```

### Install this package

Install `torch_hfaf` from this repository.

```bash
git clone https://github.com/B05901022/torch_HFAF.git
cd torch_HFAF
python setup.py install
```

## Usage

The main Mel-Spectrogram package is located under `torch_hfaf.transforms`.

```python
from torch_hfaf.transforms import AudioProcessing, GTAudioProcessing, get_default_config

config = get_default_config() # Get default configuration
feature_extractor = AudioProcessing(config)
ground_truth_extractor = GTAudioProcessing(config) # Fixed-point quantization bypassed version

audio = torch.zeros((8,63999)) # (batch, time_period*sample_rate)
mel_spec_fxp = feature_extractor(audio) # (batch, n_mels, n_frames)
mel_spec_gt = ground_truth_extractor(audio) # (batch, n_mels, n_frames)
```

To verify the result, we provided simple evaluation metrics.

```python
from torch_hfaf.metrics import mcd, mse, intel, log_intel

print(f"MCD: {mcd(mel_spec_gt, mel_spec_fxp):.4f}")
print(f"MSE: {mse(mel_spec_gt, mel_spec_fxp):.4f}")
print(f"OIM: {intel(mel_spec_gt, mel_spec_fxp):.4f}")
print(f"LOGOIM: {log_intel(mel_spec_gt, mel_spec_fxp):.4f}")
```

## Write your own configuration

Please refer to `./yaml_config/final2.yaml` for designing your own configurations. To load the `.yaml` files, simply apply the codes below

```python
import yaml

def get_config(filename: str = './yaml_config/final2.yaml'):
    with open(filename, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

config = get_config("path/to/your/config")
```

[QPyTorch-Ours]: https://github.com/B05901022/QPyTorch