
import dotenv
import hydra
import pyrootutils
import torch
from omegaconf import DictConfig

# Load environment variables from `.env` file if it exists
# Recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

# Setup root directory for imports
pyrootutils.setup_root(__file__, pythonpath=True, indicator=['README.md'])

# Disable TF32 precision for CUDA operations
torch.backends.cuda.matmul.allow_tf32 = False


@hydra.main(version_base="1.3", config_path="../../configs/", config_name="train.yaml")
def main(cfg: DictConfig) -> float:
    """
    Main training function called by Hydra.
    
    This function serves as the entry point for the training process. It imports
    necessary modules, applies optional utilities, trains the model, and returns
    the optimized metric value.
    
    Args:
        cfg (DictConfig): Configuration composed by Hydra from command line arguments
                         and config files. Contains all parameters for training.
                         
    Returns:
        float: Value of the optimized metric for hyperparameter optimization.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from gotennet import utils
    from gotennet.training_pipeline import train

    # Applies optional utilities
    utils.extras(cfg)

    # Train model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
