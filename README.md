# NH-LoRA: Future-Aware Structural Expansion of Low-Rank Adapters for Rehearsal-Free Class-Incremental Learning

This repository contains the official PyTorch implementation of NH-LoRA: Future-Aware Structural Expansion of Low-Rank Adapters for Rehearsal-Free Class-Incremental Learning.

## Overview

NH-LoRA is a rehearsal-free class-incremental learning method built on a frozen Vision Transformer and parameter-efficient low-rank adaptation. It separates reusable cross-task knowledge from task-specific expandable memory, then uses a task-aware planner to control structural decisions such as reuse, rank expansion, or new-slot creation. During learning, sparse routing and post-task consolidation help balance plasticity and stability while keeping parameter growth under control.

## Getting Started

### Environments

### Training

To train the model, you can setup the configuration in `/configs` folder first and then run it with:

```bash
scripts/<bash file benchmark you want to run>.sh
```

or in python:

```python
bash scripts/<bash file benchmark you want to run>.sh
```

### Benchmark / Dataset


#### Cifar-100

```bash
scripts/run_cifar100.sh
```

#### ImageNet-A

```bash
scripts/run_imagenet_a.sh
```

#### ImageNet-R

```bash
scripts/run_imagenet_r.sh
```

#### Custom

```bash
scripts/run_custom.sh
```

## Citation

## Acknowledgments



## License