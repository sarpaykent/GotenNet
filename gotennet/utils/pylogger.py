import logging

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
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
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
