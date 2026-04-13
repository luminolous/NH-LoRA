# NH-LoRA: Future-Aware Structural Expansion of Low-Rank Adapters for Rehearsal-Free Class-Incremental Learning

This repository contains the official PyTorch implementation of NH-LoRA: Future-Aware Structural Expansion of Low-Rank Adapters for Rehearsal-Free Class-Incremental Learning.

## Overview

NH-LoRA is a rehearsal-free class-incremental learning method built on a frozen Vision Transformer and parameter-efficient low-rank adaptation. It separates reusable cross-task knowledge from task-specific expandable memory, then uses a task-aware planner to control structural decisions such as reuse, rank expansion, or new-slot creation. During learning, sparse routing and post-task consolidation help balance plasticity and stability while keeping parameter growth under control.

## Getting Started

### Environments



### Training

To train the model, you can setup the configuration in `/configs` folder first, navigate to the main directory and then run it with:

```bash
scripts/<bash_file_benchmark_you_want_to_run>.sh
```

or in python notebook:

```python
bash scripts/<bash_file_benchmark_you_want_to_run>.sh
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

For custom training, you can use this following dataset structure:

```
data/custom-dataset/
├─ train/
│  ├─ class_000/
│  │  ├─ img_0001.jpg
│  │  ├─ img_0002.jpg
│  │  └─ ...
│  ├─ class_001/
│  ├─ class_002/
│  └─ ...
└─ test/
   ├─ class_000/
   │  ├─ img_0001.jpg
   │  └─ ...
   ├─ class_001/
   ├─ class_002/
   └─ ...
```

## Acknowledgments

This data loader and preparation implementation builds using the [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.