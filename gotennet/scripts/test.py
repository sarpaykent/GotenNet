import dotenv
import hydra
import torch
from omegaconf import DictConfig

from gotennet.utils.utils import find_config_directory  # Import the utility function

# Load environment variables from `.env` file if it exists
# Recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

# Find configs directory using the utility function
config_dir = find_config_directory()

# Disable TF32 precision for CUDA operations
torch.backends.cuda.matmul.allow_tf32 = False


@hydra.main(version_base="1.3", config_path=config_dir, config_name="test.yaml")
def main(cfg: DictConfig) -> float:
    """
    Main testing function called by Hydra.

    This function serves as the entry point for the test process. It imports
    necessary modules, applies optional utilities, trains the model, and returns
    the optimized metric value.

    Args:
        cfg (DictConfig): Configuration composed by Hydra from command line arguments
                         and config files. Contains all parameters for test.

    Returns:
        float: Value of the metric for tests.
    """
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from gotennet import utils
    from gotennet.testing_pipeline import test

    # Applies optional utilities
    utils.extras(cfg)

    # Train model
    metric_dict, _ = test(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
