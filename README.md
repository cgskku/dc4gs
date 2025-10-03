# DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting (NeurIPS 2025)<br><sub>Official Implementation</sub><br>

This repository contains the official dateset and code in the following paper:

[DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](https://cg.skku.edu/pub/2025-jeong-neurips-dc4gs)

[Moonsoo Jeong¹](https://cg.skku.edu/ppl/), [Dongbeen Kim¹](https://cg.skku.edu/ppl/), [Minseong Kim¹](https://cg.skku.edu/ppl/), [Sungkil Lee¹](https://cg.skku.edu/slee/)

¹Sungkyunkwan University

*Conference on Advances in Neural Information Processing Systems (NeurIPS) 2025*

## Overview
We present a Directional Consistency (DC)-driven Adaptive Density Control (ADC) for 3D Gaussian Splatting (DC4GS). Whereas the conventional ADC bases its primitive splitting on the magnitudes of positional gradients, we further incorporate the DC of the gradients into ADC, and realize it through the angular coherence of the gradients. Our DC better captures local structural complexities in ADC, avoiding redundant splitting. When splitting is required, we again utilize the DC to define optimal split positions so that sub-primitives best align with the local structures than the conventional random placement. As a consequence, our DC4GS greatly reduces the number of primitives (up to 30% in our experiments) than the existing ADC, and also enhances reconstruction fidelity greatly.

## Code

### Environment Setup  
Use the provided `environment.yml` file to create and install the conda environment:

```bash
conda env create -f environment.yml
conda activate DC4GS
```

### Usage  
Replace items in braces `{}` with your actual values when running the commands.

#### Training

```bash
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES={gpu_idx} \
python train-dcc.py \
  -s {dataset_dir} \
  -m {output_dir} \
  -i images or images_{scale} \
  --eval \
  --without_bound
```

- `{gpu_idx}`: GPU index to use  
- `{dataset_dir}`: Path to your dataset  
- `{output_dir}`: Directory to save checkpoints and logs  
- `images` or `images_{scale}`: Folder of original images or scaled images  

#### Rendering & Evaluation

```bash
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES={gpu_idx} \
python metrics.py \
  -m {output_dir} \
  --skip_train
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{jeong25dc4gs,
    title       = {{DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting}},
    author      = {Jeong, Moonsoo and Kim, Dongbeen and Kim, Minseong and Lee, Sungkil},
    booktitle   = {Advances in Neural Information Processing Systems (NeurIPS)},
    year        = {2025}
}
```

## Acknowledgement
This repository is based on the [3DGS](https://github.com/ykdai/Flare7K) and [AbsGS](https://github.com/TY424/AbsGS). Thanks for their awesome work.
