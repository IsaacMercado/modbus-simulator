from setuptools import setup, find_packages
from modbus_simulator import __VERSION__


def install_requires():
    with open('requirements') as reqs:
        install_req = [
            line for line in reqs.read().split('\n')
        ]
    return install_req


def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="modbus_simulator",
    url="https://github.com/riptideio/modbus-simulator.git",
    description="Modbus Simulator using Kivy, Pymodbus, Modbus-tk",
    version=__VERSION__,
    long_description=readme(),
    keywords="Modbus Simulator",
    author="riptideio",
    packages=find_packages(),
    install_requires=install_requires(),
    entry_points={
        'console_scripts': [
            'modbus.simu = modbus_simulator.main:_run',
        ],
    },
    include_package_data=True
)
