#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_cifar100.sh
bash scripts/run_imagenet_a.sh
bash scripts/run_imagenet_r.sh
