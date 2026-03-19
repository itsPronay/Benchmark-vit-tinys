import wandb

WANDB_PROJECT = "ViT Tiny"

def setup(name, config, mode):
    """Initialize a wandb run with the given name and config."""
    if mode != "disabled":
        wandb.init(
            project = WANDB_PROJECT,
            name = name,
            config = config,
        )

def log(metrics, mode):
    """Log a dictionary of metrics to the current wandb run."""
    if mode != "disabled":
        wandb.log(metrics)
        wandb.finish()