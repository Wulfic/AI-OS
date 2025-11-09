#!/usr/bin/env python3
"""
Simple DDP Launcher for AI-OS Multi-GPU Training
Launches training across multiple GPUs using torch.distributed.

Usage:
    python scripts/simple_ddp_launcher.py --nproc_per_node=2 aios.cli.aios hrm-hf train-actv1 --ddp <args>
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description='Simple DDP Launcher for AI-OS')
    parser.add_argument('--nproc_per_node', type=int, required=True,
                        help='Number of processes (GPUs) per node')
    parser.add_argument('module', help='Python module to run (e.g., aios.cli.aios)')
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help='Arguments to pass to the module')
    return parser.parse_args()


def run_worker(rank, world_size, module, module_args):
    """Worker function that runs on each GPU"""
    # Set up environment variables for distributed training
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Set CUDA device for this process
    torch.cuda.set_device(rank)
    
    # Import and run the module
    import runpy
    sys.argv = [module] + module_args
    
    try:
        runpy.run_module(module, run_name='__main__')
    except SystemExit as e:
        if e.code != 0:
            raise


def main():
    args = parse_args()
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    available_gpus = torch.cuda.device_count()
    if args.nproc_per_node > available_gpus:
        print(f"ERROR: Requested {args.nproc_per_node} GPUs but only {available_gpus} available!")
        sys.exit(1)
    
    print(f"Launching DDP training across {args.nproc_per_node} GPUs...")
    print(f"Module: {args.module}")
    print(f"Args: {' '.join(args.args)}")
    print("")
    
    # Launch processes
    mp.spawn(
        run_worker,
        args=(args.nproc_per_node, args.module, args.args),
        nprocs=args.nproc_per_node,
        join=True
    )
    
    print("\nDDP training completed!")


if __name__ == '__main__':
    main()
