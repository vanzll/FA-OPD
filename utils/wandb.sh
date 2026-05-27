#! /usr/bin/env bash

conf=${1}
result=$(python -m wandb sweep ${conf} 2>&1)
sweep=$(echo $result | sed "s/^.*wandb agent \([^[:space:]]*\).*$/\1/")
echo "wandb agent ${sweep}"
python -m wandb agent ${sweep}
