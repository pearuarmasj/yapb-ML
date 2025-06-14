#!/usr/bin/env python3
"""Quick GPU test script"""
import torch

print("=== GPU STATUS CHECK ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("🚀 READY FOR BEAST MODE!")
else:
    print("❌ CUDA not available - need to install GPU PyTorch")
    print("Current PyTorch is CPU-only")
