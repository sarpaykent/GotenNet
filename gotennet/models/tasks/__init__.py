"""Task implementations for various molecular datasets."""

from __future__ import absolute_import, division, print_function

from gotennet.models.tasks.QM9Task import QM9Task

# Dictionary mapping task names to their implementations
TASK_DICT = {
    'QM9': QM9Task,      # QM9 quantum chemistry dataset
}
