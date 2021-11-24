from pathlib import Path
from distutils.version import LooseVersion
import platform

IS_DARWIN = platform.system().lower() == "darwin"
OSX_SIERRA = LooseVersion("10.12")

if IS_DARWIN:
    IS_HIGH_SIERRA_OR_ABOVE = LooseVersion(platform.mac_ver()[0])
else:
    IS_HIGH_SIERRA_OR_ABOVE = False

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