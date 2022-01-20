"""'
Modbus Simu App
===============
"""
import re
from os import path
from struct import error as StructError
import json
from pathlib import Path
from configparser import ConfigParser
from typing import Callable

from kivy.app import App
from kivy.properties import (
    ObjectProperty,
    ListProperty,
    BooleanProperty
)
from kivy.animation import Animation
from kivy.config import Config

from kivy.uix.textinput import TextInput
from kivy.uix.settings import SettingsWithSidebar
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.label import Label
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.splitter import Splitter
from kivy.uix.actionbar import ActionButton, ActionPrevious
from kivy.uix.tabbedpanel import TabbedPanel

from serial.serialutil import SerialException

from modbus_simulator.utils.constants import BLOCK_TYPES
from modbus_simulator.utils.common import configure_modbus_logger
from modbus_simulator.utils.background_job import BackgroundJob
from modbus_simulator.utils.modbus import ModbusSimulator
from modbus_simulator.ui.settings import SettingIntegerWithRange
from modbus_simulator.ui.datamodel import DataModel

from .conts import ASSETS_DIR, TEMPLATES_DIR, DEFAULT_SERIAL_PORT, MAP

ROOT = Path(__file__).parent
SLAVES_FILE = ROOT.joinpath("slaves.json")
settings_icon = str(ASSETS_DIR.joinpath("Control-Panel.png"))
app_icon = str(ASSETS_DIR.joinpath("riptideLogo.png"))

with open(ROOT.joinpath('config.json'), 'r') as file:
    setting_panel = file.read()


class SelectableRecycleBoxLayout(FocusBehavior,
                                 LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    pass


class SelectableLabel(RecycleDataViewBehavior, Label):
    index = None
    selected: bool = BooleanProperty(False)
    selectable: bool = BooleanProperty(True)
    callback_selected: Callable = ObjectProperty()

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        rv.data[index]['selected'] = self.selected = is_selected

    def on_selected(self, label: Label, is_select: bool):
        if self.callback_selected:
            self.callback_selected(label, is_select)


class SelectableRecycleView(RecycleView):

    @property
    def selection(self):
        return list(
            filter(
                lambda item: item.get('selected'),
                self.data
            )
        )

    def index(self, key: str):
        for i, k in enumerate(self.data):
            if k['text'] == key:
                return i
        return -1


class Gui(BoxLayout):
    """
    Gui of widgets. This is the root widget of the app.
    """

    # ---------------------GUI------------------------ #
    # Checkbox to select between tcp/serial
    interfaces: BoxLayout = ObjectProperty()

    tcp: CheckBox = ObjectProperty()
    serial: CheckBox = ObjectProperty()

    # Boxlayout to hold interface settings
    interface_settings: BoxLayout = ObjectProperty()

    # TCP port
    port: TextInput = ObjectProperty()

    # Toggle button to start/stop modbus server
    start_stop_server: ToggleButton = ObjectProperty()

    # Container for slave list
    slave_pane: Splitter = ObjectProperty()
    # slave start address textbox
    slave_start_add: TextInput = ObjectProperty()
    # slave end address textbox
    slave_end_add: TextInput = ObjectProperty()
    # Slave device count text box
    slave_count: TextInput = ObjectProperty()
    # Slave list
    slave_list: SelectableRecycleView = ObjectProperty()

    # Container for modbus data models
    data_model_loc: BoxLayout = ObjectProperty()
    # Tabbed panel to hold various modbus datamodels
    data_models: TabbedPanel = ObjectProperty()

    # Data models
    data_count: TextInput = ObjectProperty()
    data_model_coil: DataModel = ObjectProperty()
    data_model_discrete_inputs: DataModel = ObjectProperty()
    data_model_input_registers: DataModel = ObjectProperty()
    data_model_holding_registers: DataModel = ObjectProperty()

    settings: ActionButton = ObjectProperty()
    riptide_logo: ActionPrevious = ObjectProperty()

    reset_sim_btn: ActionButton = ObjectProperty()

    # Helpers
    # slaves = ["%s" %i for i in range(1, 248)]
    _data_map = {"tcp": {}, "rtu": {}}
    active_slave = None
    server_running = False
    simulating = False
    simu_time_interval = None
    anim = None
    restart_simu = False
    sync_modbus_thread = None
    sync_modbus_time_interval = 5
    _modbus_device = {"tcp": None, 'rtu': None}
    _slaves = {"tcp": None, "rtu": None}

    last_active_port = {"tcp": "", "serial": ""}
    active_server = "tcp"
    _serial_settings_changed = False

    config: ConfigParser

    def __init__(self, time_interval=1, modbus_log=None, **kwargs):
        super(Gui, self).__init__(**kwargs)
        self.settings.icon = settings_icon
        self.riptide_logo.app_icon = app_icon
        self.config = Config.get_configparser('app')
        self.data_model_loc.disabled = True
        self.slave_pane.disabled = True
        self._init_coils()
        self._init_registers()
        self._register_config_change_callback(
            self._update_serial_connection,
            'Modbus Serial'
        )
        self.data_model_loc.disabled = True
        cfg = {
            'no_modbus_log': ~self.config.getboolean("Logging", "logging"),
            'no_modbus_console_log': ~self.config.getboolean("Logging", "console logging"),
            'modbus_console_log_level': self.config.get("Logging", "console log level"),
            'modbus_file_log_level': self.config.get("Logging", "file log level"),
            'no_modbus_file_log': self.config.getboolean("Logging", "file logging"),
            'modbus_log': modbus_log
        }
        configure_modbus_logger(cfg, protocol_logger="pymodbus")
        self.simu_time_interval = time_interval
        self.sync_modbus_thread = BackgroundJob(
            "modbus_sync",
            self.sync_modbus_time_interval,
            self._sync_modbus_block_values
        )
        self.sync_modbus_thread.start()
        self._slave_misc = {
            "tcp": [
                self.slave_start_add.text,
                self.slave_end_add.text,
                self.slave_count.text
            ],
            "rtu": [
                self.slave_start_add.text,
                self.slave_end_add.text,
                self.slave_count.text
            ]
        }

    @property
    def modbus_device(self) -> ModbusSimulator:
        return self._modbus_device[self.active_server]

    @modbus_device.setter
    def modbus_device(self, value):
        self._modbus_device[self.active_server] = value

    @property
    def slave(self):
        return self._slaves[self.active_server]

    @slave.setter
    def slave(self, value):
        self._slaves[self.active_server] = value

    @property
    def data_map(self):
        return self._data_map[self.active_server]

    @data_map.setter
    def data_map(self, value):
        self._data_map[self.active_server] = value

    def _init_coils(self):
        time_interval = self.config.getint("Simulation", "time interval")
        minval = self.config.getint("Modbus Protocol", "bin min")
        maxval = self.config.getint("Modbus Protocol", "bin max")

        self.data_model_coil.init(
            blockname="coils",
            simulate=self.simulating,
            time_interval=time_interval,
            minval=minval,
            maxval=maxval,
            _parent=self
        )
        self.data_model_discrete_inputs.init(
            blockname="discrete_inputs",
            simulate=self.simulating,
            time_interval=time_interval,
            minval=minval,
            maxval=maxval,
            _parent=self
        )

    def _init_registers(self):
        time_interval = self.config.getint("Simulation", "time interval")
        minval = self.config.getint("Modbus Protocol", "reg min")
        maxval = self.config.getint("Modbus Protocol", "reg max")
        self.block_start = self.config.getint("Modbus Protocol", "block start")
        self.block_size = self.config.getint("Modbus Protocol", "block size")
        self.word_order = self.config.get("Modbus Protocol", "word order")
        self.byte_order = self.config.get("Modbus Protocol", "byte order")

        self.data_model_input_registers.init(
            blockname="input_registers",
            simulate=self.simulating,
            time_interval=time_interval,
            minval=minval,
            maxval=maxval,
            _parent=self
        )
        self.data_model_holding_registers.init(
            blockname="holding_registers",
            simulate=self.simulating,
            time_interval=time_interval,
            minval=minval,
            maxval=maxval,
            _parent=self
        )

    def _register_config_change_callback(self, callback, section, key=None):
        self.config.add_callback(callback, section, key)

    def _update_serial_connection(self, *args):
        self._serial_settings_changed = True

    def _create_modbus_device(self):
        kwargs = {}
        create_new = False
        kwargs['byte_order'] = self.byte_order
        kwargs['word_order'] = self.word_order

        if self.active_server == "rtu":
            kwargs["baudrate"] = self.config.getint(
                'Modbus Serial', "baudrate")
            kwargs["bytesize"] = self.config.getint(
                'Modbus Serial', "bytesize")
            kwargs["parity"] = self.config.get(
                'Modbus Serial', "parity")
            kwargs["stopbits"] = self.config.getint(
                'Modbus Serial', "stopbits")
            kwargs["xonxoff"] = self.config.getboolean(
                'Modbus Serial', "xonxoff")
            kwargs["rtscts"] = self.config.getboolean(
                'Modbus Serial', "rtscts")
            kwargs["dsrdtr"] = self.config.getboolean(
                'Modbus Serial', "dsrdtr")
            kwargs["writetimeout"] = self.config.getint(
                'Modbus Serial', "writetimeout")
            kwargs["timeout"] = self.config.getboolean(
                'Modbus Serial', "timeout")
        elif self.active_server == 'tcp':
            kwargs['address'] = self.config.get('Modbus Tcp', 'ip')

        if not self.modbus_device:
            create_new = True
        else:
            if self.modbus_device.server_type == self.active_server:
                if (str(self.modbus_device.port) != str(self.port.text)) or self._serial_settings_changed:
                    create_new = True
            else:
                create_new = True
        if create_new:

            self.modbus_device = ModbusSimulator(
                server=self.active_server,
                port=self.port.text,
                **kwargs
            )

            if self.slave is None:
                self.slave = []

            self._serial_settings_changed = False

    def start_server(self, button: ToggleButton):
        if button.state == "down":
            try:
                self._start_server()
            except SerialException as err:
                button.state = "normal"
                self.show_error("Error in opening Serial port: %s" % err)
                return
            button.text = "Stop"
        else:
            self._stop_server()
            button.text = "Start"

    def _start_server(self):
        self._create_modbus_device()

        self.modbus_device.start()
        self.server_running = True
        self.interface_settings.disabled = True
        self.interfaces.disabled = True
        self.slave_pane.disabled = False

        if len(self.slave_list.selection):
            self.data_model_loc.disabled = False
            if self.simulating:
                self._simulate()

    def _stop_server(self):
        self.simulating = False
        self._simulate()
        self.modbus_device.stop()
        self.server_running = False
        self.interface_settings.disabled = False
        self.interfaces.disabled = False
        self.slave_pane.disabled = True
        self.data_model_loc.disabled = True

    def update_tcp_connection_info(self, checkbox, value):
        self.active_server = "tcp"
        if value:
            self.interface_settings.current = checkbox
            if self.last_active_port['tcp'] == "":
                self.last_active_port['tcp'] = 5440
            self.port.text = self.last_active_port['tcp']
            self._restore()
        else:
            self.last_active_port['tcp'] = self.port.text
            self._backup()

    def update_serial_connection_info(self, checkbox, value):
        self.active_server = "rtu"
        if value:
            self.interface_settings.current = checkbox
            if self.last_active_port['serial'] == "":
                self.last_active_port['serial'] = DEFAULT_SERIAL_PORT
            self.port.text = self.last_active_port['serial']
            self._restore()
        else:
            self.last_active_port['serial'] = self.port.text
            self._backup()

    def show_error(self, e):
        self.info_label.text = str(e)
        self.anim = Animation(top=190.0, opacity=1, d=2, t='in_back') +\
            Animation(top=190.0, d=3) +\
            Animation(top=0, opacity=0, d=2)
        self.anim.start(self.info_label)

    def add_slaves(self, *args):
        selected = self.slave_list.selection
        data = self.slave_list.data
        ret = self._process_slave_data(data)
        self._add_slaves(selected, data, ret)

    def _add_slaves(self, selected, data, ret):
        if ret[0]:
            start_slave_add, slave_count = ret[1:]
        else:
            return

        for slave_to_add in range(
            start_slave_add, start_slave_add + slave_count
        ):
            if str(slave_to_add) in self.data_map:
                return
            self.data_map[str(slave_to_add)] = {
                "coils": {
                    'data': {},
                    'item_strings': [],
                    "instance": self.data_model_coil,
                    "dirty": False
                },
                "discrete_inputs": {
                    'data': {},
                    'item_strings': [],
                    "instance": self.data_model_discrete_inputs,
                    "dirty": False
                },
                "input_registers": {
                    'data': {},
                    'item_strings': [],
                    "instance": self.data_model_input_registers,
                    "dirty": False
                },
                "holding_registers": {
                    'data': {},
                    'item_strings': [],
                    "instance": self.data_model_holding_registers,
                    "dirty": False
                }
            }
            self.modbus_device.add_slave(slave_to_add)
            for block_name, block_type in BLOCK_TYPES.items():
                self.modbus_device.add_block(
                    slave_to_add,
                    block_name, block_type,
                    self.block_start,
                    self.block_size
                )

            data.append({
                'text': str(slave_to_add),
                'selected': False,
                'callback_selected': self.select_slave
            })
        self.slave_list.data = data
        # self.slave_list._trigger_reset_populate()

        for item in selected:
            index = self.slave_list.index(item['text'])
            view = self.slave_list.view_adapter.get_index(index)
            if not view.is_selected:
                self.view.trigger_action(duration=0)

        self.slave_start_add.text = str(start_slave_add + slave_count)
        self.slave_end_add.text = self.slave_start_add.text
        self.slave_count.text = "1"

    def _process_slave_data(self, data):
        success = True
        data = sorted(data, key=lambda i: int(i['text']))
        starting_address = int(self.slave_start_add.text)
        end_address = int(self.slave_end_add.text)

        if end_address < starting_address:
            end_address = starting_address

        try:
            slave_count = int(self.slave_count.text)
        except ValueError:
            slave_count = 1

        if str(starting_address) in data:
            self.show_error(
                "slave already present (%s)" % starting_address
            )
            success = False
            return [success]
        if starting_address < 1:
            self.show_error(
                "slave address (%s)"
                " should be greater than 0 " % starting_address
            )
            success = False
            return [success]
        if starting_address > 247:
            self.show_error(
                "slave address (%s)"
                " beyond supported modbus slave "
                "device address (247)" % starting_address
            )
            success = False
            return [success]

        size = (end_address - starting_address) + 1
        size = slave_count if slave_count > size else size

        if (size + starting_address) > 247:
            self.show_error(
                "address range (%s) beyond "
                "allowed modbus slave "
                "devices(247)" % (size + starting_address)
            )
            success = False
            return [success]

        self.slave_end_add.text = str(starting_address + size - 1)
        self.slave_count.text = str(size)

        return success, starting_address, size

    def delete_slaves(self, *args):
        selected = self.slave_list.selection
        slave = self.active_slave
        current_tab = self.data_models.current_tab

        for item in selected:
            text = item['text']
            self.modbus_device.remove_slave(int(text))
            self.slave_list.data.pop(self.slave_list.index(text))
            current_tab.content.clear_widgets(make_dirty=True)

            if self.simulating:
                self.simulating = False
                self.restart_simu = True
                self._simulate()

            self.data_map.pop(slave)

    def update_data_models(self, *args):
        active = self.active_slave
        tab = self.data_models.current_tab
        count = int(self.data_count.text)
        value = {}
        for i in range(count):
            _value = {'value': 1}
            if tab in ['input_registers', 'holding_registers']:
                _value['formatter'] = 'uint16'

            value[i] = _value

        self._update_data_models(active, tab, value)

    def _update_data_models(self, active, tab, value):
        ct = tab
        current_tab = MAP[ct.text]

        ct.content.update_view()
        _data = self.data_map[active][current_tab]
        registers = sum(
            map(
                lambda val: int(
                    ''.join(list(filter(
                        str.isdigit, str(val.get('formatter', '16')))))
                ), _data['data'].values()))/16

        # Old schema
        if isinstance(value, list):
            _new_value = list(value)
            value = {}
            for index, v in enumerate(_new_value):
                if not isinstance(v, dict):
                    value[index] = {'value': v}
                if current_tab in ['input_registers', 'holding_registers']:
                    value[index]['formatter'] = 'uint16'
        if registers+len(value) <= self.block_size:
            list_data, item_strings = ct.content.add_data(value)
            _data['item_strings'].extend(item_strings)
            _data['item_strings'] = list(set(_data['item_strings']))
            _data['data'].update(list_data)
            self.update_backend(int(active), current_tab, list_data)
        else:
            msg = (
                "OutOfModbusBlockError: address %s"
                " is out of block size %s" % (
                    len(value),
                    self.block_size
                )
            )
            self.show_error(msg)

    def sync_data_callback(self, blockname, data):
        ct = self.data_models.current_tab
        current_tab = MAP[ct.text]
        if blockname != current_tab:
            current_tab = blockname
        try:
            _data = self.data_map[self.active_slave][current_tab]
            _data['data'].update(data)
            for k, v in data.items():
                # v = v if not isinstance(v, dict) else v['value']
                if blockname in ['holding_registers', 'input_registers']:
                    self.modbus_device.encode(
                        int(self.active_slave),
                        current_tab,
                        k,
                        float(v['value']),
                        v['formatter']
                    )
                else:
                    # v = dict(value=int(v))
                    if not isinstance(v, dict):
                        v = dict(value=v)
                    self.modbus_device.set_values(
                        int(self.active_slave),
                        current_tab,
                        k,
                        v.get('value')
                    )
        except KeyError:
            pass
        except StructError:
            self.show_error("Invalid value supplied , Check the formatter!")

    def sync_formatter_callback(self, blockname, data, old_formatter):
        ct = self.data_models.current_tab
        current_tab = MAP[ct.text]
        if blockname != current_tab:
            current_tab = blockname
        try:
            _data = self.data_map[self.active_slave][current_tab]
            _updated = {}
            current = list(data.keys())
            for k in current:
                old_wc = int(''.join(list(
                    filter(str.isdigit, str(old_formatter))
                )))/16
                new_wc = int(''.join(list(
                    filter(str.isdigit, data[k].get('formatter'))
                )))/16
                new_val, count = self.modbus_device.decode(
                    int(self.active_slave),
                    current_tab, k, data[k]['formatter']
                )
                data[k]['value'] = new_val
                _updated['offset'] = k
                _updated['count'] = count
                if old_wc > new_wc:
                    missing = self.modbus_device.get_values(
                        int(self.active_slave),
                        current_tab, int(k) + new_wc,
                        old_wc-new_wc
                    )
                    for i, val in enumerate(missing):
                        o = int(k) + new_wc + i
                        o = int(o)
                        if not isinstance(k, int):
                            o = str(o)
                        data[o] = {'value': val, 'formatter': 'uint16'}
            _data['data'].update(data)
            _data['data'] = dict(ct.content.update_registers(
                _data['data'],
                _updated
            ))

        except KeyError:
            pass

    def delete_data_entry(self, *args):
        ct = self.data_models.current_tab
        current_tab = MAP[ct.text]
        _data = self.data_map[self.active_slave][current_tab]
        item_strings = _data['item_strings']
        deleted, data = ct.content.delete_data(item_strings)
        dm = _data['data']

        for index in deleted:
            dm.pop(index, None)

        if deleted:
            self.update_backend(int(self.active_slave), current_tab, data)
            self.show_error((
                "Deleting individual modbus register/discrete_inputs/coils "
                "is not supported. The data is removed from GUI and "
                "the corresponding value is updated to '0' in backend . "
            ))

    def select_slave(self, label: Label, is_selected: bool):
        ct = self.data_models.current_tab

        if len(self.slave_list.selection) != 1:
            # Multiple selection - No Data Update
            ct.content.clear_widgets(make_dirty=True)

            if self.simulating:
                self.simulating = False
                self.restart_simu = True
                self._simulate()

            self.data_model_loc.disabled = True
            self.active_slave = None

        else:
            self.data_model_loc.disabled = False

            if self.restart_simu:
                self.simulating = True
                self.restart_simu = False
                self._simulate()

            self.active_slave = self.slave_list.selection[0]['text']
            self.refresh()

    def refresh(self):
        for child in self.data_models.tab_list:
            dm = self.data_map[self.active_slave][MAP[child.text]]['data']
            # child.content.refresh(dm)

    def update_backend(self, slave_id, blockname, new_data):
        self.modbus_device.remove_block(slave_id, blockname)
        self.modbus_device.add_block(
            slave_id,
            blockname,
            BLOCK_TYPES[blockname], 0,
            self.block_size
        )

        for data in new_data:
            if blockname in ['holding_registers', 'input_registers']:
                self.modbus_device.encode(
                    slave_id,
                    blockname,
                    int(data['offset']),
                    float(data['value']),
                    data['formatter']
                )
            else:
                self.modbus_device.set_values(
                    slave_id,
                    blockname,
                    int(data['offset']),
                    int(data['value'])
                )

    def change_simulation_settings(self, **kwargs):
        self.data_model_coil.reinit(**kwargs)
        self.data_model_discrete_inputs.reinit(**kwargs)
        self.data_model_input_registers.reinit(**kwargs)
        self.data_model_holding_registers.reinit(**kwargs)

    def change_datamodel_settings(self, key: str, value: str):
        if "max" in key:
            data = {"maxval": float(value)}
        else:
            data = {"minval": float(value)}

        if "bin" in key:
            self.data_model_coil.reinit(**data)
            self.data_model_discrete_inputs.reinit(**data)
        else:
            self.data_model_input_registers.reinit(**data)
            self.data_model_holding_registers.reinit(**data)

    def start_stop_simulation(self, button: ToggleButton):
        if button.state == "down":
            self.simulating = True
            self.reset_sim_btn.disabled = True
        else:
            self.simulating = False
            self.reset_sim_btn.disabled = False
            if self.restart_simu:
                self.restart_simu = False
        self._simulate()

    def _simulate(self):
        self.data_model_coil.start_stop_simulation(self.simulating)
        self.data_model_discrete_inputs.start_stop_simulation(self.simulating)
        self.data_model_input_registers.start_stop_simulation(self.simulating)
        self.data_model_holding_registers.start_stop_simulation(
            self.simulating)

    def reset_simulation(self, *args):
        if not self.simulating:
            self.data_model_coil.reset_block_values()
            self.data_model_discrete_inputs.reset_block_values()
            self.data_model_input_registers.reset_block_values()
            self.data_model_holding_registers.reset_block_values()

    def _sync_modbus_block_values(self):
        """
        track external changes in modbus block values and sync GUI
        ToDo:
        A better way to update GUI when simulation is on going  !!
        """
        if not self.simulating:
            if self.active_slave:
                _data_map = self.data_map[self.active_slave]
                for block_name, value in _data_map.items():
                    updated = {}
                    for k, v in value['data'].items():
                        if block_name in ['input_registers', 'holding_registers']:
                            actual_data, count = self.modbus_device.decode(
                                int(self.active_slave), block_name, k,
                                v['formatter']
                            )
                        else:
                            actual_data = self.modbus_device.get_values(
                                int(self.active_slave),
                                block_name,
                                int(k),

                            )
                            actual_data = actual_data[0]
                        try:
                            if actual_data != float(v['value']):
                                v['value'] = actual_data
                                updated[k] = v
                        except TypeError:
                            pass
                    if updated:
                        value['data'].update(updated)
                        self.refresh()

    def _backup(self):
        if self.slave is not None:
            self.slave = self.slave_list.data
        self._slave_misc[self.active_server] = [
            self.slave_start_add.text,
            self.slave_end_add.text,
            self.slave_count.text
        ]

    def _restore(self):
        if self.slave is None:
            self.slave = []

        self.slave_list.data = self.slave
        (
            self.slave_start_add.text,
            self.slave_end_add.text,
            self.slave_count.text
        ) = self._slave_misc[self.active_server]
        # self.slave_list._trigger_reset_populate()

    def save_state(self):
        with open(SLAVES_FILE, 'w') as f:
            slave = [int(slave_no.get('text'))
                     for slave_no in self.slave_list.data]
            slaves_memory = []
            for slaves, mem in self.data_map.items():
                for name, value in mem.items():
                    if len(value['data']) != 0:
                        slaves_memory.append(
                            (slaves, name, value['data'])
                        )

            json.dump(dict(
                slaves_list=slave, active_server=self.active_server,
                port=self.port.text, slaves_memory=slaves_memory
            ), f, indent=4)

    def load_state(self):
        if not self.config.getboolean("State", "load state") or not SLAVES_FILE.is_file():
            return

        with open(SLAVES_FILE, 'r') as file:
            try:
                data = json.load(file)
            except ValueError as error:
                self.show_error(
                    "LoadError: Failed to load previous simulation state : %s " % error.message
                )
                return

            if (
                'active_server' not in data
                    or 'port' not in data
                    or 'slaves_list' not in data
                    or 'slaves_memory' not in data
            ):
                self.show_error(
                    "LoadError: Failed to load previous "
                    "simulation state : JSON Key Missing"
                )
                return

            slaves_list = data['slaves_list']
            if not len(slaves_list):
                return

            if data['active_server'] == 'tcp':
                self.tcp.active = True
                self.serial.active = False
                self.interface_settings.current = self.tcp
            else:
                self.tcp.active = False
                self.serial.active = True
                self.interface_settings.current = self.serial

            self.active_server = data['active_server']
            self.port.text = data['port']
            self.word_order = self.config.get("Modbus Protocol", "word order")
            self.byte_order = self.config.get("Modbus Protocol", "byte order")
            self._create_modbus_device()

            start_slave = 0
            temp_list = []
            slave_count = 1
            for first, second in zip(slaves_list[:-1], slaves_list[1:]):
                if first+1 == second:
                    slave_count += 1
                else:
                    temp_list.append((slaves_list[start_slave], slave_count))
                    start_slave += slave_count
                    slave_count = 1
            temp_list.append((slaves_list[start_slave], slave_count))

            for start_slave, slave_count in temp_list:
                self._add_slaves(
                    self.slave_list.selection,
                    self.slave_list.data,
                    (True, start_slave, slave_count)
                )

            memory_map = {
                'coils': self.data_models.tab_list[3],
                'discrete_inputs': self.data_models.tab_list[2],
                'input_registers': self.data_models.tab_list[1],
                'holding_registers': self.data_models.tab_list[0]
            }
            slaves_memory = data['slaves_memory']
            for slave_memory in slaves_memory:
                active_slave, memory_type, memory_data = slave_memory
                _data = self.data_map[active_slave][memory_type]
                _data['data'].update(memory_data)
                _data['item_strings'] = list(sorted(memory_data.keys()))
                self.update_backend(
                    int(active_slave),
                    memory_type,
                    memory_data
                )
                # self._update_data_models(
                #    active_slave,
                #    memory_map[memory_type],
                #    memory_data
                # )


class ModbusSimulatorApp(App):
    '''
    The kivy App that runs the main root. All we do is build a Gui
    widget into the root.
    '''
    gui = None
    title = "Modbus Simulator"
    settings_cls = None
    use_kivy_settings = True
    settings_cls = SettingsWithSidebar
    kv_directory = TEMPLATES_DIR.as_posix()

    @property
    def modbus_log_dir(self):
        return path.join(self.user_data_dir, 'modbus.log')

    def build(self):
        self.gui = Gui(modbus_log=self.modbus_log_dir)
        self.gui.load_state()
        return self.gui

    def on_pause(self):
        return True

    def on_stop(self):
        if self.gui.server_running:
            if self.gui.simulating:
                self.gui.simulating = False
                self.gui._simulate()
            self.gui.modbus_device.stop()
        self.gui.sync_modbus_thread.cancel()
        self.config.write()
        self.gui.save_state()

    def show_settings(self, btn):
        self.open_settings()

    def build_config(self, config: ConfigParser):
        config.add_section('Modbus Tcp')
        config.add_section('Modbus Protocol')
        config.add_section('Modbus Serial')
        config.set('Modbus Tcp', "ip", '127.0.0.1')
        config.set('Modbus Protocol', "block start", 0)
        config.set('Modbus Protocol', "block size", 100)
        config.set('Modbus Protocol', "byte order", 'big')
        config.set('Modbus Protocol', "word order", 'big')
        config.set('Modbus Protocol', "bin min", 0)
        config.set('Modbus Protocol', "bin max", 1)
        config.set('Modbus Protocol', "reg min", 0)
        config.set('Modbus Protocol', "reg max", 65535)
        config.set('Modbus Serial', "baudrate", 9600)
        config.set('Modbus Serial', "bytesize", "8")
        config.set('Modbus Serial', "parity", 'N')
        config.set('Modbus Serial', "stopbits", "1")
        config.set('Modbus Serial', "xonxoff", 0)
        config.set('Modbus Serial', "rtscts", 0)
        config.set('Modbus Serial', "dsrdtr", 0)
        config.set('Modbus Serial', "writetimeout", 2)
        config.set('Modbus Serial', "timeout", 2)

        config.add_section('Logging')
        config.set('Logging', "log file",  self.modbus_log_dir)

        config.set('Logging', "logging", 1)
        config.set('Logging', "console logging", 1)
        config.set('Logging', "console log level", "DEBUG")
        config.set('Logging', "file log level", "DEBUG")
        config.set('Logging', "file logging", 1)

        config.add_section('Simulation')
        config.set('Simulation', 'time interval', 1)

        config.add_section('State')
        config.set('State', 'load state', 1)

    def build_settings(self, settings: SettingsWithSidebar):
        settings.register_type("numeric_range", SettingIntegerWithRange)
        settings.add_json_panel(
            'Modbus Settings',
            self.config,
            data=setting_panel
        )

    def on_config_change(self, config: ConfigParser, section: str, key: str, value: str):
        if config is not self.config:
            return
        if section == "Simulation" and key == "time interval":
            self.gui.change_simulation_settings(time_interval=eval(value))
        if section == "Modbus Protocol" and key in (
            "bin max", "bin min", "reg max", "reg min", "override", "word order", "byte order"
        ):
            self.gui.change_datamodel_settings(key, value)
        if section == "Modbus Protocol" and key == "block start":
            self.gui.block_start = int(value)
        if section == "Modbus Protocol" and key == "block size":
            self.gui.block_size = int(value)

    def close_settings(self, *args):
        super(ModbusSimulatorApp, self).close_settings()
