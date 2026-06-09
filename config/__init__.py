"""Application configuration.

Modules may keep using ``import config as cfg`` while settings are grouped by
responsibility in this package.
"""

from .data import *
from .exchanges import *
from .execution import *
from .features import *
from .labeling import *
from .paths import *
from .trading import *
from .training import *

__all__ = [
    name
    for name in globals()
    if name.isupper() or name == "effective_max_label_horizon"
]
