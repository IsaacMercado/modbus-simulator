'''
Modbus Simulator App
====================
'''
import argparse
import os

parser = argparse.ArgumentParser('Use pymodbus as modbus backend')

os.environ.setdefault('KIVY_NO_ARGS', '1')


def main():
    from modbus_simulator.ui import run
    run()


if __name__ == "__main__":
    parser.parse_args()
    main()
