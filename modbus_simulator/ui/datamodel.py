from random import randint, uniform
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
# from kivy.adapters.dictadapter import DictAdapter
# from kivy.uix.listview import ListItemButton, CompositeListItem, ListView, SelectableView
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown

from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview import RecycleView


from modbus_simulator.utils.background_job import BackgroundJob
from .conts import TEMPLATES_DIR

Builder.load_file(str(TEMPLATES_DIR.joinpath('datamodel.kv')))
integers_dict = {}

Data = List[Dict[str, Union[str, int, str]]]


class ErrorPopup(Popup):
    """
    Popup class to display error messages
    """

    def __init__(self, title, text, **kwargs):
        super(ErrorPopup, self).__init__()
        # super(ErrorPopup, self).__init__(**kwargs)

        content = BoxLayout(orientation="vertical")
        content.add_widget(Label(text=text, font_size=20))

        button = Button(text="Dismiss", size_hint=(1, .20), font_size=20)
        button.bind(on_release=lambda *args: self.dismiss())
        content.add_widget(button)

        self.content = content
        self.title = title
        self.auto_dismiss = False
        self.size_hint = .7, .5
        self.font_size = 20

        self.open()


class ListItemReprMixin(Label):
    """
    repr class for ListItem Composite class
    """

    def __repr__(self):
        if isinstance(self.text, str):
            text = self.text.encode('utf-8')
        else:
            text = self.text
        return '<%s text=%s>' % (self.__class__.__name__, text)

# init reinit start_stop_simulation reset_block_values


class RowDataLayout(RecycleDataViewBehavior, BoxLayout):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    offset = NumericProperty()
    value = NumericProperty()
    formatter: str = StringProperty()

    offset_label: Label = ObjectProperty()
    value_input: TextInput = ObjectProperty()
    formatter_dropdown: DropDown = ObjectProperty()

    def refresh_view_attrs(self, rv: RecycleView, index: int, data: List[Dict[str, Any]]):
        self.index = index
        return super(RowDataLayout, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        if super(RowDataLayout, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv: RecycleView, index: int, is_selected: bool):
        rv.data[index]['selected'] = self.selected = is_selected

    def on_value(self, instance, value):
        pass

    def on_formatter(self, instance, value):
        pass


class DataModel(RecycleView):
    """
    Uses :class:`CompositeListItem` for list item views comprised by two
    :class:`ListItemButton`s and one :class:`ListItemLabel`. Illustrates how
    to construct the fairly involved args_converter used with
    :class:`CompositeListItem`.
    """
    minval: float = NumericProperty(0)
    maxval: float = NumericProperty(0)

    simulate = False
    time_interval = 1
    dirty_thread = False
    dirty_model = False
    simulate_timer = None
    simulate = False
    dispatcher = None
    list_view = None
    _parent = None
    is_simulating = False
    blockname = "<BLOCK_NAME_NOT_SET>"

    formatter: str = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init()

    def init(self, simulate: bool = False, time_interval: float = 1, **kwargs):
        """
        Initializes Datamodel

        """
        self.minval = kwargs.get("minval", self.minval)
        self.maxval = kwargs.get("maxval", self.maxval)
        self.blockname = kwargs.get("blockname", self.blockname)
        self.clear_widgets()
        self.simulate = simulate
        self.time_interval = time_interval

        '''
        dict_adapter = DictAdapter(
            data={},
            args_converter=self.arg_converter,
            selection_mode='single',
            allow_empty_selection=True,
            cls=RecycleDataViewBehavior
        )

        # Use the adapter in our ListView:
        self.list_view = ListView(adapter=dict_adapter)
        self.add_widget(self.list_view)
        '''

        self.dispatcher = UpdateEventDispatcher()
        self._parent = kwargs.get('_parent', None)
        """self.simulate_timer = BackgroundJob(
            "simulation",
            self.time_interval,
            self._simulate_block_values
        )"""

    def clear_widgets(self, children=None, make_dirty=False):
        """
        Overidden Clear widget function used while deselecting/deleting slave
        """
        if make_dirty:
            self.dirty_model = True
        return super().clear_widgets(children=children)

    def reinit(self, **kwargs):
        """
        Re-initializes Datamodel on change in model configuration from settings
        """
        self.minval = kwargs.get("minval", self.minval)
        self.maxval = kwargs.get("maxval", self.maxval)
        time_interval = kwargs.get("time_interval", None)
        try:
            if time_interval and int(time_interval) != self.time_interval:
                self.time_interval = time_interval
                if self.is_simulating:
                    self.simulate_timer.cancel()
                self.simulate_timer = BackgroundJob(
                    "simulation",
                    self.time_interval,
                    self._simulate_block_values
                )
                self.dirty_thread = False
                self.start_stop_simulation(self.simulate)
        except ValueError:
            Logger.debug("Error while reinitializing DataModel %s" % kwargs)

    def update_view(self) -> None:
        """
        Updates view with listview again
        """
        if self.dirty_model:
            # self.add_widget(self.list_view)
            self.dirty_model = False

    def get_address(self, offset, as_string=False) -> Union[str, int]:
        offset = int(offset)

        if self.blockname == "coils":
            offset = offset
        elif self.blockname == "discrete_inputs":
            offset = 10001 + offset if offset < 10001 else offset
        elif self.blockname == "input_registers":
            offset = 30001 + offset if offset < 30001 else offset
        else:
            offset = 40001 + offset if offset < 40001 else offset

        return str(offset) if as_string else offset

    def arg_converter(self, index, data):
        """
        arg converter to convert data to list view
        :param index:
        :param data:
        :return:
        """
        _id = self.get_address(self.sorted_keys[index])

        payload = {
            'text': str(_id),
            'size_hint_y': None,
            'height': 30,
            'cls_dicts': [
                {
                    # 'cls': ListItemButton,
                    'cls': Button,
                    'kwargs': {'text': str(_id)}
                }
            ]
        }
        if self.blockname in ['input_registers', 'holding_registers']:
            payload['cls_dicts'].extend([
                {
                    'cls': NumericTextInput,
                    'kwargs': {
                        'data_model': self,
                        'minval': self.minval,
                        'maxval': self.maxval,
                        'text': str(data['value']),
                        'multiline': False,
                        'is_representing_cls': True,
                    }
                },
                {
                    'cls': DropBut,
                    'kwargs': {
                        'data_model': self,
                        'text': data.get('formatter', 'uint16')
                    }
                }
            ]
            )
        else:
            payload['cls_dicts'].append(
                {
                    'cls': NumericTextInput,
                    'kwargs': {
                        'data_model': self,
                        'minval': self.minval,
                        'maxval': self.maxval,
                        'text': str(data['value']),
                        'multiline': False,
                        'is_representing_cls': True,

                    }
                }
            )

        return payload

    @property
    def sorted_keys(self):
        return sorted(map(
            lambda row: int(row['address']),
            self.data
        ))

    def add_data(self, data):
        """
        Adds data to the Data model
        :param data:
        :param item_strings:
        :return:
        """
        item_strings = []
        self.update_view()
        current_keys = self.sorted_keys
        next_index = 0

        if current_keys:
            next_index = int(max(current_keys)) + 1

        data = [{
            'offset': self.get_address(int(offset) + next_index),
            **data
        } for offset, data in data.items()]

        for index in range(len(data)):
            # offset = self.get_address(offset)
            offset = data[index]['offset']
            item_strings.append(offset)

            if int(offset) >= 30001:
                if not data[index].get('formatter'):
                    data[index]['formatter'] = 'uint16'

        # self.list_view.adapter.data.update(data)
        self.data = data

        return self.data, item_strings

    def delete_data(self, item_strings):
        """
        Delete data from data model
        :param item_strings:
        :return:
        """
        selections = self.list_view.adapter.selection
        items_popped = []

        for item in selections:
            index_popped = item_strings.pop(item_strings.index(int(item.text)))
            self.list_view.adapter.data.pop(int(item.text), None)
            self.list_view.adapter.update_for_new_data()
            self.list_view._trigger_reset_populate()
            items_popped.append(index_popped)

        return items_popped, self.data

    def on_data_update(self, index, data):
        """
        Call back function to update data when data is changed in the list view
        :param index:
        :param data:
        :return:
        """
        index = self.get_address(self.list_view.adapter.sorted_keys[index])

        try:
            self.data[index]
        except KeyError:
            index = str(index)

        if self.blockname in ['input_registers', 'holding_registers']:
            self.data[index]['value'] = float(data)
        else:
            self.data[index]['value'] = int(data)

        data = {
            'event': 'sync_data',
            'data': {index: self.data[index]}
        }
        self.dispatcher.dispatch(
            'on_update',
            self._parent,
            self.blockname,
            data
        )

    def on_formatter_update(self, index: int, old: str, new: str) -> None:
        """
        Callback function to use the formatter selected in the list view
        """
        index = self.get_address(int(self.data[index]['address']))

        try:
            self.data[index]['formatter'] = new
        except KeyError:
            index = str(index)
            self.data[index]['formatter'] = new

        _data = {
            'event': 'sync_formatter',
            'old_formatter': old,
            'data': {index: self.data[index]}
        }

        self.dispatcher.dispatch(
            'on_update',
            self._parent,
            self.blockname,
            _data
        )

    def update_registers(self, new_values, update_info: Dict):
        offset = update_info.get('offset')
        count = update_info.get('count')
        to_remove = None

        if count > 1:
            to_remove = range(offset+1, offset+count)

        self.refresh(new_values, to_remove)
        return self.data

    def refresh(self, data: Data = [], to_remove: Optional[Iterable] = None):
        """
        Data model refresh function to update when the view when slave is
        selected
        :param data:
        :param to_remove:
        :return:
        """
        self.update_view()

        if not data or len(data) != len(self.data):
            self.data = data
        else:
            for index, idata in enumerate(data):
                if self.data[index] != idata:
                    self.data[index] = idata

        if to_remove:
            for entry in sorted(to_remove, reverse=True):
                self.data.pop(entry, None)

        self.disabled = False

    def start_stop_simulation(self, simulate):
        """
        Starts or stops simulating data
        :param simulate:
        :return:
        """
        self.simulate = simulate

        if self.simulate:
            if self.dirty_thread:
                self.simulate_timer = BackgroundJob(
                    "simulation",
                    self.time_interval,
                    self._simulate_block_values
                )
            self.simulate_timer.start()
            self.dirty_thread = False
            self.is_simulating = True
        else:
            self.simulate_timer.cancel()
            self.dirty_thread = True
            self.is_simulating = False

    def _simulate_block_values(self):
        if self.simulate and self.data:
            for index, value in enumerate(self.data):
                formatter = self.data[index]['formatter']

                if self.blockname in ['input_registers', 'holding_registers']:
                    if 'float' in formatter:
                        value = round(uniform(self.minval, self.maxval), 2)
                    else:
                        value = randint(self.minval, self.maxval)
                        if 'uint' in formatter:
                            value = abs(value)
                else:
                    value = randint(self.minval, self.maxval)

                data = self.data[index]
                data['value'] = value
                self.data[index] = data

            self.refresh(self.data)

            data = {'event': 'sync_data', 'data': self.data}
            self.dispatcher.dispatch(
                'on_update',
                self._parent,
                self.blockname,
                data
            )

    def reset_block_values(self):
        if not self.simulate:
            if self.data:
                for index, data in enumerate(self.data):
                    data['value'] = 1
                    self.data[index] = data

                self.disabled = False
                self._parent.sync_data_callback(
                    self.blockname,
                    self.data
                )


class DropBut(DropDown):
    # drop_list = None
    types = [
        'int16',
        'int32',
        'int64',
        'uint16',
        'uint32',
        'uint64',
        'float32',
        'float64'
    ]
    drop_down = None

    def __init__(self, data_model: DataModel, **kwargs):
        super(DropBut, self).__init__(**kwargs)
        self.data_model = data_model
        self.drop_down = DropDown()

        for t in self.types:
            button = Button(
                text=t, size_hint_y=None, height=45,
                background_color=(0.0, 0.5, 1.0, 1.0)
            )
            button.bind(on_release=lambda b: self.drop_down.select(b.text))
            self.drop_down.add_widget(button)

        self.bind(on_release=self.drop_down.open)
        self.drop_down.bind(on_select=self.on_formatter_select)

    def select_from_composite(self, *args):
        # self.bold = True
        pass

    def deselect_from_composite(self, *args):
        # self.bold = False
        pass

    def on_formatter_select(self, instance, value):
        self.data_model.on_formatter_update(self.index, self.text, value)
        self.text = value


class NumericTextInput(TextInput):
    """
    :class:`~kivy.uix.listview.NumericTextInput` mixes
    :class:`~kivy.uix.listview.SelectableView` with
    :class:`~kivy.uix.label.TextInput` to produce a label suitable for use in
    :class:`~kivy.uix.listview.ListView`.
    """
    edit: bool = BooleanProperty(False)

    def __init__(self, data_model: DataModel, minval: float, maxval: float, **kwargs):
        self.minval = minval
        self.maxval = maxval
        self.data_model = data_model
        super(NumericTextInput, self).__init__(**kwargs)
        try:
            self.val = int(self.text)
        except ValueError:
            self.hint_text = "Only numeric value in range {0}-{1} to be used".format(
                minval,
                maxval
            )

        self._update_width()
        self.disabled = True

    def _update_width(self):
        if self.data_model.blockname not in ['input_registers', 'holding_registers']:
            self.padding_x = self.width

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and not self.edit:
            self.edit = True
            self.select()
        return super(NumericTextInput, self).on_touch_down(touch)

    def select(self, *args):
        self.disabled = False
        self.bold = True
        if isinstance(self.parent, RecycleDataViewBehavior):
            for child in self.parent.children:
                # print child.children
                pass
            self.parent.select_from_child(self, *args)

    def deselect(self, *args):
        self.bold = False
        self.disabled = True
        if isinstance(self.parent, RecycleDataViewBehavior):
            self.parent.deselect_from_child(self, *args)

    def select_from_composite(self, *args):
        self.bold = True

    def deselect_from_composite(self, *args):
        self.bold = False

    def on_text_validate(self, *args):
        try:
            float(self.text)

            if not(self.minval <= float(self.text) <= self.maxval):
                raise ValueError
            self.edit = False
            self.data_model.on_data_update(self.index, self.text)
            self.deselect()
        except ValueError:
            error_text = f"Only numeric value in range {self.minval}-{self.maxval} to be used"
            ErrorPopup(title="Error", text=error_text)
            self.text = ""
            self.hint_text = error_text
            return

    def on_text_focus(self, instance, focus):
        if focus is False:
            self.text = instance.text
            self.edit = False
            self.deselect()


class UpdateEventDispatcher(EventDispatcher):
    '''
    Event dispatcher for updates in Data Model
    '''

    def __init__(self, **kwargs):
        self.register_event_type('on_update')
        super(UpdateEventDispatcher, self).__init__(**kwargs)

    def on_update(self, _parent, blockname, data):
        Logger.debug(
            "In UpdateEventDispatcher "
            "on_update {parent:%s, "
            "blockname: %s, data:%s,}" % (
                _parent,
                blockname,
                data
            )
        )
        event = data.pop('event', None)
        if event == 'sync_data':
            _parent.sync_data_callback(
                blockname,
                data.get('data', {})
            )
        else:
            old_formatter = data.pop("old_formatter", None)
            _parent.sync_formatter_callback(
                blockname,
                data.get('data', {}),
                old_formatter
            )
