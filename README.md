# HETT

## Introduction

The official repository for AAAI 2026 oral paper [History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation](https://arxiv.org/abs/2512.14222).

Aerial Vision-and-Language Navigation (AVLN) requires Unmanned Aerial Vehicle (UAV) agents to localize targets in large-scale urban environments based on linguistic instructions. While successful navigation demands both global environmental reasoning and local scene comprehension, existing UAV agents typically adopt mono-granularity frameworks that struggle to balance these two aspects. To address this limitation, this work proposes a History-Enhanced Two-Stage Transformer (HETT) framework, which integrates the two aspects through a coarse-to-fine navigation pipeline. Specifically, HETT first predicts coarse-grained target positions by fusing spatial landmarks and historical context, then refines actions via fine-grained visual analysis. In addition, a historical grid map is designed to dynamically aggregate visual features into a structured spatial memory, enhancing comprehensive scene awareness. Additionally, the CityNav dataset annotations are manually refined to enhance data quality. Experiments on the refined CityNav dataset show that HETT delivers significant performance gains, while extensive ablation studies further verify the effectiveness of each component.



Project Page:[HETT](https://crotonyl.github.io/HETT.git.io/)



## Setup

This code was developed with Python 3.10, PyTorch 2.2.2, and CUDA 11.3 on Ubuntu 22.04.

To set up the environment, create the conda environment and install PyTorch.

```bash
conda create -n hett python=3.10 &&
conda activate hett &&
conda install pytorch torchvision pytorch-cuda=11.3 -c pytorch -c nvidia
```

Install the dependencies for HETT.

```bash
pip install -r requirements.txt
```

## Data Preparation

Follow the instruction in [CityNav](https://github.com/water-cookie/citynav) for data download.

Download the refined dataset and corresponding checkpoints:

https://www.dropbox.com/scl/fo/ua3kn6mw3adn2hcsdikt3/AEdoRoo-OeXnbrpeOpk4BcQ?rlkey=v5rlqiqa1k9ml8isinpv7glgk&st=ahseq0r2&dl=0

## Usage

```bash
cd multiagent
# train
./train.sh
# eval
./eval.sh
```

## Citation

```bibtex
@misc{ding2025historyenhancedtwostagetransformeraerial,
      title={History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation}, 
      author={Xichen Ding and Jianzhe Gao and Cong Pan and Wenguan Wang and Jie Qin},
      year={2025},
      eprint={2512.14222},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.14222}, 
}
```

## Acknowledgements

We would like to express our gratitude to the authors of the following codebase.

- [CityNav](https://github.com/water-cookie/citynav)
- [AVDN](https://github.com/eric-ai-lab/Aerial-Vision-and-Dialog-Navigation)