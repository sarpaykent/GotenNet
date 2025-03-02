"""  """

from __future__ import print_function, absolute_import, division

from src.models.tasks.MD17Task import MD17Task
# from src.models.tasks.Molecule3DTask import Molecule3DTask
from src.models.tasks.QM9Task import QM9Task

TASK_DICT = {
    'QM9': QM9Task,
    'MD17': MD17Task,
}
