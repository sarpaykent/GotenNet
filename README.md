# GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks
**[Paper](https://openreview.net/pdf?id=5wxCQDtbMo)**  |  **[Project](https://www.sarpaykent.com/publications/gotennet/)** 

This is the official implementation of "GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks". 

## Overview

GotenNet is a novel framework for modeling 3D molecular structures that balances expressiveness and efficiency through:

- Leverages effective geometric tensor representations without relying on irreducible representations or Clebsch-Gordan transforms
- Introduces unified structural embedding with geometry-aware tensor attention
- Implements hierarchical tensor refinement for flexible and efficient representations
- Achieves state-of-the-art performance on QM9, rMD17, MD22, and Molecule3D datasets

<p align="center"> <img src="assets/GotenNet_framework.png" width="800"> </p> 

## Installation

```bash
# Create and activate conda environment
conda create -n gotennet python=3.10
conda activate gotennet

# Install requirements
pip install -r requirements.txt
```

## Project Structure

```
├── configs/               <- Configuration files
│   ├── data/             <- Dataset configs
│   ├── model/            <- Model architecture configs
│   ├── train.yaml        <- Main training config
│   └── eval.yaml         <- Evaluation config
│
├── src/                  <- Source code
│   ├── data/            <- Data processing
│   ├── models/          <- Model implementation
│   └── utils/           <- Utility functions
│
├── scripts/              <- Training and evaluation scripts
└── requirements.txt      <- Python dependencies
```

## Training

Train the model on QM9 dataset for U0 target prediction:

```bash
python src/train.py experiment=gotennet_u0
```

## Results

Our model achieves state-of-the-art performance across multiple benchmarks:

- QM9: Superior performance in both scalar and vector property prediction
- rMD17: Improved force field predictions
- MD22: Enhanced molecular dynamics simulation accuracy
- Molecule3D: Strong performance in 3D structure prediction

## Acknowledgements

GotenNet is proudly built on the innovative foundations provided by the projects below.
- [e3nn](https://github.com/e3nn/e3nn)
- [PyG](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)

## Citation

Please consider citing our work below if this project is helpful:


```bibtex
@inproceedings{aykent2025rethinking,
  author = {Aykent, Sarp and Xia, Tian},
  booktitle = {The {Thirteenth} {International} {Conference} on {Learning} {Representations}},
  year = {2025},
  title = {GotenNet: Rethinking {Efficient} 3D {Equivariant} {Graph} {Neural} {Networks}},
  url = {https://openreview.net/forum?id=5wxCQDtbMo},
  howpublished = {https://openreview.net/forum?id=5wxCQDtbMo},
}
```