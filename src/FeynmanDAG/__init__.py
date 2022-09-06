"""
Copyright (c) 2022 Uwe Hernandez Acosta. All rights reserved.

FeynmanDAG: Simple Feynman diagram generation
"""


from __future__ import annotations

from ._version import version as __version__

__all__ = [
    "__version__",
]


from . import particles
from .particles import *  # noqa

__all__ += particles.__all__

from . import interaction  # noqa
from .interaction import *  # noqa

__all__ += interaction.__all__

from . import diagrams  # noqa
from .diagrams import *  # noqa

__all__ += diagrams.__all__


from . import algorithm  # noqa
from .algorithm import *  # noqa

__all__ += algorithm.__all__

from . import kinematics  # noqa
