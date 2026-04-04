#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_cifar100.sh
bash scripts/run_cub200.sh
bash scripts/run_imagenet_r.sh
bash scripts/run_omnibenchmark.sh
