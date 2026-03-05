"""Task implementations for various molecular datasets."""

from __future__ import absolute_import, division, print_function

from gotennet.models.tasks.QM9Task import QM9Task
from gotennet.models.tasks.MDTask import MDTask

# Dictionary mapping task names to their implementations
TASK_DICT = {
    'QM9': QM9Task,      # QM9 quantum chemistry dataset
    'rMD17': MDTask,    # Revised MD17 dataset
}
