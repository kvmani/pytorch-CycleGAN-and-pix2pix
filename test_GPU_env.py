#!/usr/bin/env python3
import sys
import os
import platform
import subprocess
import datetime

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return out.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"Error running {cmd}: {e.output.decode().strip()}"

def main():
    print("=== Environment Check Report ===")
    print("Date:", datetime.datetime.now())
    print("OS:", platform.platform())
    print("Python:", sys.version.replace("\n", " "))
    print("PyTorch version:", getattr(__import__('torch'), '__version__', 'not installed'))
    if 'torch' in sys.modules or True:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA version used by torch:", torch.version.cuda)
            print("Number of GPUs detected:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}:", torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                print(f"    Capability:", props.major, props.minor)
                print(f"    Total memory (GB):", round(props.total_memory / (1024**3), 2))
    # Show nvidia-smi summary
    print("\n=== NVIDIA-SMI Info ===")
    print(run("nvidia-smi"))

    # Show environmental CUDA variables
    print("\n=== CUDA Environment Variables ===")
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES"):
        print(f"{var}: {os.environ.get(var, '')}")

    print("\n=== Pip Packages (torch, torchvision) ===")
    print(run("pip show torch torchvision"))

if __name__ == "__main__":
    main()
