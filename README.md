# ViT TinyS Benchmark Suite

This project runs benchmarks for the following models on GPU, CPU, and edge devices:
- vit_tiny_patch16_224
- mobilevitv2_100
- mobilevitv2_125
- tiny_vit_5m_224
- vitfs_tiny_patch16_gap_reg4_dinov2_bn_init
- vitfs_tiny_patch16_gap_reg4_dinov2_init


### Command-Line Arguments

| Argument           | Description |
|--------------------|-------------|
| --runs             | Number of runs to perform for benchmarking (default: 10) |
| --warmup_runs      | Number of warmup runs before benchmarking (default: 10) |
| --models           | List of model names to benchmark (default: all supported models) |
| --image_sizes      | List of image sizes to run the benchmark on (default: 224) |
| --device           | Device to run the benchmark on: cpu, cuda (GPU), ai_hub (edge), or all. (default: cpu) |
| --ai_hub_device    | Name of the physical AI Hub device to use (default: Samsung Galaxy S25 (Family)) |
| --wandb_mode       | WandB logging mode: online, offline, or disabled (default: online) |


## How to Run


1. **Run the benchmark**
   - To run the benchmark with default settings:
     ```bash
     python run.py
     ```
   - To specify options (e.g., device, models, image sizes):
     ```bash
     python run.py --device cpu --models mobilevitv2_100 --image_sizes 224 
     ```
  - You can also pass multiple arguments for models and image_size
    ```bash
     python run.py --device cpu --models vit_tiny_patch16_224 mobilevitv2_100 --image_sizes 224 448
    ```
    

3. **Results**
   - Benchmark results will be saved to `benchmark_results.csv` in the project directory.

## Note
- The script supports benchmarking on GPU, CPU, and edge devices (Qualcomm AI Hub).
- You need to setup qualcom ai hub and W&B (Optional)

## How to set up Qualcomm AI Hub and Weights & Biases (wandb)

- **Qualcomm AI Hub:**
  - Install the SDK:
    ```bash
    pip3 install qai-hub
    ```
  - Configure with your API token (get it from your Qualcomm AI Hub Settings page):
    ```bash
    qai-hub configure --api_token API_TOKEN
    ```

- **Weights & Biases (wandb):**

  - You must have a wandb account. From user settings page, get your API_KEY
  - Log in to wandb:
    ```python
    pip install wandb
    import wandb
    wandb.login()
    ```

- For edge device benchmarking, configure the `--ai_hub_device` argument as needed.

