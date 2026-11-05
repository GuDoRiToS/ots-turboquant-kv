# Omni-Scaled Fast Vector Quantization for LLM KV Cache Compression (OTS-TQ-Fast)
Official Python/NumPy/PyTorch implementation for "Omni-Scaled Fast Vector Quantization for LLM KV Cache Compression: Resolving Token Norm Imbalance with Unbiased Residual Projections".
## Overview
OTS-TQ-Fast bridges OScaR (Omni-Token Scaling & Canalized Rotation) and TurboQuant (PolarQuant Lloyd-Max Beta quantization + 1-bit Quantized Johnson-Lindenstrauss residual projection) to eliminate Token Norm Imbalance (TNI) in low-bit LLM Key-Value (KV) caches without offline calibration data or custom CUDA kernels.
