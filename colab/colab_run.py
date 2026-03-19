import argparse
import wandb
import colab
import timm
import torch
import pandas as pd
import torch
import time
import numpy as np


def run(model, device, image_size, runs=10, warmup_runs=10):
    
    model = model.to(device)
    input_tensor = torch.randn(1, 3, image_size, image_size).to(device)
    metrics = benchmark_model(model, input_tensor=input_tensor, runs=runs, warmup_runs=warmup_runs)
    
    return metrics


def benchmark_model(model, input_tensor, runs, warmup_runs, run_warmup=True):

  model.eval()

  #warmup
  with torch.no_grad():
    if run_warmup:
      for _ in range(warmup_runs):
        _ = model(input_tensor)

  #sync before timing (run on gpu only)
  if input_tensor.device.type == 'cuda':
    torch.cuda.synchronize()

  latencies = []

  for _ in range(runs):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_tensor)
    if input_tensor.device.type == 'cuda':
        torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)

  mean_latency_ms = float(np.mean(latencies))
  min_latency_ms = float(np.min(latencies))
  max_latency_ms = float(np.max(latencies))
  std = float(np.std(latencies))

  result = {
      'estimated_inference_time_ms' : 'NOT_AVAILABLE',
      'mean_latency_ms' : round(mean_latency_ms, 4),
      'min_latency_ms' : round(min_latency_ms, 4),
      'max_latency_ms' : round(max_latency_ms, 4),
      'std_latency_ms' : round(std, 4),
  }

  return result
