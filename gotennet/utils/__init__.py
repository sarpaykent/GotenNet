import logging
import warnings
from typing import Sequence

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def humanbytes(B):
    """
    Return the given bytes as a human friendly KB, MB, GB, or TB string.
    
    Args:
        B: Number of bytes.
        
    Returns:
        str: Human-readable string representation of bytes.
    """
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


from gotennet.utils.logging_utils import log_hyperparameters as log_hyperparameters
from gotennet.utils.pylogger import get_pylogger as get_pylogger
from gotennet.utils.utils import get_metric_value as get_metric_value
from gotennet.utils.utils import task_wrapper as task_wrapper


def get_logger(name=__name__) -> logging.Logger:
    """
    Initialize multi-GPU-friendly python command line logger.
    
    Args:
        name: Name of the logger, defaults to the module name.
        
    Returns:
        logging.Logger: Logger instance with rank zero only decorators.
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """
    Apply optional utilities, controlled by config flags.
    
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    
    Args:
        config: DictConfig containing the hydra config.
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
        config: DictConfig,
        print_order: Sequence[str] = (
                "datamodule",
                "model",
                "callbacks",
                "logger",
                "trainer",
        ),
        resolve: bool = True,
) -> None:
    """
    Print content of DictConfig using Rich library and its tree structure.
    
    Args:
        config: Configuration composed by Hydra.
        print_order: Determines in what order config components are printed.
            Defaults to ("datamodule", "model", "callbacks", "logger", "trainer").
        resolve: Whether to resolve reference fields of DictConfig. Defaults to True.
    """
    import rich.syntax
    import rich.tree
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)
