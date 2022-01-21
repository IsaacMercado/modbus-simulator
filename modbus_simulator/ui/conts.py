from pathlib import Path
from distutils.version import LooseVersion
from typing import Dict, Tuple, Union
import platform


IS_DARWIN = platform.system().lower() == "darwin"
OSX_SIERRA = LooseVersion("10.12")
IS_HIGH_SIERRA_OR_ABOVE = LooseVersion(
    platform.mac_ver()[0]
) if IS_DARWIN else False
DEFAULT_SERIAL_PORT = '/dev/ptyp0' if not IS_HIGH_SIERRA_OR_ABOVE else '/dev/ttyp0'

MAP = {
    "coils": "coils",
    'discrete inputs': 'discrete_inputs',
    'input registers': 'input_registers',
    'holding registers': 'holding_registers'
}

BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR.joinpath('assets')
TEMPLATES_DIR = BASE_DIR.joinpath('templates')

DEFAULT_VALUE = 1
DEFAULT_FORMATTER = "uint16"

TYPE_RANGES: Dict[str, Tuple[int]] = {
    'boolean': (0, 1),
    'int16': (-32768, 32767),
    'int32': (-2147483648, 2147483647),
    'int64': (-9223372036854775808, 9223372036854775807),
    'uint16': (0, 65535),
    'uint32': (0, 4294967295),
    'uint64': (0, 18446744073709551615),
    'float32': (-3.4028235e+38, 3.4028235e+38),
    'float64': (-1.7976931348623157e+308, 1.7976931348623157e+308)
}

Number = Union[int, float]
Data = Dict[str, Union[Number, str]]
