"""
Energy Indicators package
"""

from importlib.metadata import version, PackageNotFoundError

from .wind import *
from .solar import *
from .aggregation import *
from .run_energy_indicators import *
from .plot import *
from .mask_processing import *

try:
    __version__ = version("energy_indicators")
except PackageNotFoundError:
    __version__ = "unknown (package not installed)"