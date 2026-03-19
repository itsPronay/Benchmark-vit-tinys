import numpy as np


def us_to_ms(x):
    return x / 1e3

def bytes_to_mb(x):
    return x / (1024 ** 2)

def extract_metrics_from_profile(profile: dict):
    exec_sum    = profile.get("execution_summary", {})
    
    # ── Latency Metrics ────────────────────────────────────
    times = np.array(exec_sum.get("all_inference_times", []))

    metrics = {
        "estimated_inference_time_ms": round(us_to_ms(exec_sum.get("estimated_inference_time", 0)), 4),
        
        "mean_latency_ms":    round(us_to_ms(times.mean()), 4)             if len(times) else None,
        "min_latency_ms":     round(us_to_ms(times.min()), 4)              if len(times) else None,
        "max_latency_ms":     round(us_to_ms(times.max()), 4)              if len(times) else None,
        "std_latency_ms":     round(us_to_ms(times.std()), 4)              if len(times) else None,
    }

    return metrics