"""
TexJEPA specialization launcher.

This entry point dispatches to ``src.pretrain_v2.main`` and enables the
texture-aware extensions used by TexJEPA-N, TexJEPA-R, and TexJEPA-C:

* ``data.context_noise`` for context-only texture corruption.
* ``meta.num_registers`` and ``*_reg`` ViT factories for register tokens.
* ``optimization.vicreg`` for patch-token variance/covariance regularization.

Use ``main_pretrain.py`` for the common 200-epoch I-JEPA base run, then use
this launcher for the 50-epoch TexJEPA specialization stages.
"""

import argparse
import multiprocessing as mp
import os
import pprint
import signal
import sys
import yaml

from src.utils.distributed import init_distributed
from src.pretrain_v2 import main as app_main

# ── NCCL robustness — identical to main_pretrain.py ──
os.environ.setdefault('NCCL_TIMEOUT', '1800')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_P2P_LEVEL', 'NVL')
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/pretrain/texjepa_n.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)


_children: list[mp.Process] = []


def _sigterm_handler(signum, frame):
    for p in _children:
        if p.is_alive():
            os.kill(p.pid, signal.SIGTERM)
    for p in _children:
        p.join(timeout=30)
    sys.exit(1)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)

    if num_gpus == 1:
        process_main(0, args.fname, 1, args.devices)
    else:
        mp.set_start_method('spawn')
        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGINT, _sigterm_handler)
        for rank in range(num_gpus):
            p = mp.Process(
                target=process_main,
                args=(rank, args.fname, num_gpus, args.devices),
            )
            p.start()
            _children.append(p)
        exit_code = 0
        for p in _children:
            p.join()
            if p.exitcode != 0:
                exit_code = p.exitcode
        sys.exit(exit_code)
