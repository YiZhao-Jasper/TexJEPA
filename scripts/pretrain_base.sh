#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
DEVICES="${DEVICES:-cuda:0 cuda:1}"
CONFIG="${CONFIG:-configs/pretrain/ijepa_300.yaml}"

exec ${PYTHON_BIN} -u main_pretrain.py --fname "${CONFIG}" --devices ${DEVICES}
